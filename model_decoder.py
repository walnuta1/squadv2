"""
A question and answer decoder model.
"""
import tensorflow as tf

import utils
import transformer

def squad_v2_decoder(
        sequence_output,
        input_mask,
        segment_ids,
        embedding_size=768,
        dropout_prob=0.1,
        initializer_range=0.2
    ):
    """
    Question and answer decoder that, based on the combined
    embedding vector sequence of the paragraph text and the
    question, returns a tuple
        (is_answerable, start_pos, end_pos).

    Args:
        sequence_output: the sequence of vector embeddings
                         from the nlp_encoder.
        input_mask: the mask indicating valid input tokens.
        segment_ids: the segment ids. Zero for question
                     tokens and one for text tokens.
        embedding_size: the size of embedding vectors.
        dropout_prob: the dropout probability.
        initializer_range: the range for normal truncated init.
    Returns:
        A touple of (is_answerable, start_pos, end_pos).
    """
    adder = (1.0 - tf.cast(input_mask, tf.float32)) * \
            (1.0 - tf.cast(segment_ids, tf.float32)) * -10000.0
    batch_size = tf.shape(sequence_output)[0] # use dynamic shape for batch_size

    # First get the pooler output
    with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        pooled_output = tf.squeeze(sequence_output[:, 0:1, :], axis=1)

    # Predict answerability from the pooler output
    with tf.variable_scope("answerable"):
        answerable_weights = tf.get_variable(
            "weights", [1, 1, embedding_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        answerable_weights_expanded = tf.add(
            answerable_weights,
            tf.zeros([batch_size, 1, embedding_size], dtype=tf.float32)
        )
        answerable_attn, _ = transformer.multiheaded_attention_no_transform(
            answerable_weights_expanded, sequence_output, sequence_output, None,
            hidden_size=embedding_size, head_count=1
        )
        answerable_vector = tf.reshape(answerable_attn, [batch_size, embedding_size])
        answerable_logits = tf.layers.dense(
            answerable_vector,
            2,
            activation=utils.get_activation("relu"),
            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
        )

    # From the pooler output, predict a vector whose dot products with the sequence
    # produce the start positional logits
    with tf.variable_scope("start_pos"):
        start_pos_vector = tf.layers.dense(
            utils.dropout(tf.concat([pooled_output, answerable_vector], axis=-1),
                          dropout_prob=dropout_prob),
            embedding_size,
            activation=utils.get_activation("relu"),
            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
        )
        start_pos_vector_expand = tf.expand_dims(start_pos_vector, axis=1)
        start_pos_logits = tf.matmul(start_pos_vector_expand, sequence_output, transpose_b=True)
        start_pos_logits = tf.squeeze(start_pos_logits, axis=[1])
        start_pos_logits += adder

    # From the pooler output, and the start position vector, produce a end position
    # vector. The new vector dot proucts with the sequence to produce the end
    # position logits
    with tf.variable_scope("end_pos"):
        end_pos_vector = tf.layers.dense(
            utils.dropout(tf.concat([pooled_output, answerable_vector, start_pos_vector], axis=-1),
                          dropout_prob=dropout_prob),
            embedding_size,
            activation=utils.get_activation("relu"),
            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
        )
        end_pos_vector_expand = tf.expand_dims(end_pos_vector, axis=1)
        end_pos_logits = tf.matmul(end_pos_vector_expand, sequence_output, transpose_b=True)
        end_pos_logits = tf.squeeze(end_pos_logits, axis=[1])
        end_pos_logits += adder

    return answerable_logits, start_pos_logits, end_pos_logits
