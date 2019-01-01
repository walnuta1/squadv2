"""
  Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder. The implementation
  style here mimic that of the PyTorch designs.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  http://nlp.seas.harvard.edu/2018/04/03/attention.html
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  https://github.com/google-research/bert/blob/master/modeling.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import utils

def multiheaded_attention(
        query, key, value, mask, \
        hidden_size=512, \
        head_count=12, \
        dropout_prob=0.1, \
        initializer_range=0.02 \
        ):
    """
    This is an implementation of multiheaded attention mechanism described
    in the "Attention is all you need" paper. In the transform model, key
    and value parameter are usually the same. In the case of self-attention,
    query parameter is also the same. But query, key and value are all first
    linearly projected so that each of the "head_count" attentions can attend
    to different parts of a given sequence.

    The shape of query, key & value is (batch_size, seq_length, model_size).
    query and key can have different seq_length while key and value are
    assumed to have the same sequence_length.

    Args:
    query, key, value: the respective attention vectors.
    mask: of shape [batch_size, seq_length, seq_length].
    embedding_size: the dimension of the input tensor embedding
    hidden_size: the size of the model after projection for attention
    head_count: the number of parallel attention count.
    dropout_prob: the dropout probabililyt to apply for attention.
    initializer_range: normal initialization std constant for linear projections
    """
    query_shape = query.shape.as_list()
    key_shape = key.shape.as_list()
    value_shape = value.shape.as_list()
    batch_size = query_shape[0]
    query_seq_len = query_shape[1]
    key_seq_len = key_shape[1]
    embedding_size = value_shape[2]
    assert hidden_size % head_count == 0
    size_per_head = hidden_size // head_count

    # Perform linear projections
    query_projected = tf.layers.dense(
        utils.reshape_to_matrix(query),
        hidden_size,
        name="query",
        kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
    )
    key_projected = tf.layers.dense(
        utils.reshape_to_matrix(key),
        hidden_size,
        name="key",
        kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
    )
    value_projected = tf.layers.dense(
        utils.reshape_to_matrix(value),
        hidden_size,
        name="value",
        kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
    )

    # Transform tensors to be of shape in order:
    # (batch_size, head_count, size_per_head, per_head_model_size)
    def transform_for_multihead(input_tensor, batch_size,
                                num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    query_transformed = transform_for_multihead(
        query_projected, batch_size, head_count, query_seq_len, size_per_head)
    key_transformed = transform_for_multihead(
        key_projected, batch_size, head_count, key_seq_len, size_per_head)
    value_transformed = transform_for_multihead(
        value_projected, batch_size, head_count, key_seq_len, size_per_head)

    # Compute scaled dot product
    attention_scores = tf.matmul(
        query_transformed, key_transformed, transpose_b=True)
    attention_scores = tf.multiply(
        attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    # Apply mask
    if mask is not None:
        # Add a dimension after batch_size to represent heads
        mask = tf.expand_dims(mask, axis=[1])
        # The original masks have 1 for valid positions and 0 for
        # invalid positions, the transformed adder would have
        # value 0 for valie position and -10000 for invalid positions.
        adder = (1.0 - tf.cast(mask, tf.float32)) * (-10000.0)
        attention_scores += adder

    # Normalize attention scores to probabilities, perform dropout
    attention_prob = tf.nn.softmax(attention_scores)
    attention_prob = utils.dropout(attention_prob, dropout_prob=dropout_prob)

    # Compute the attended values and transform it back to
    # the original shape
    value_after_attention = tf.matmul(attention_prob, value_transformed)
    value_after_attention_orig = tf.transpose(
        value_after_attention, [0, 2, 1, 3])
    value_after_attention_orig = tf.reshape(
        value_after_attention_orig, [batch_size, query_seq_len, hidden_size])

    # return the result with a final linear projection, along with
    # attention probabilities
    return tf.layers.dense(
        value_after_attention_orig,
        embedding_size,
        name="projection",
        kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
    ), attention_prob

def transformer_encoder(input_tensor,
                        attention_mask=None,
                        hidden_size=768,
                        num_encoder_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        intermediate_act_fn=utils.gelu,
                        dropout_prob=0.1,
                        initializer_range=0.02):
    """
    Multi-headed multi-layer transfomer encoder implementation

    Args:
        input_tensor: of shape (batch_size, seq_length, embedding_size)
        attention_mask: of shape (batch_size, seq_length, seq_length)
        hidden_size: the dimension in multiheaded attention after projection
        num_encoder_layers: number of transformer encoder layers
        num_attention_heads: number of attention heads
        intermediate_size: the intermediate size in the linear
                           thransformation layer in encoders
        intermediate_act_fn: the activation function for the
                             encoder linear transformations
        dropout_prob: the dropout probability
        intializer_range: the clipped normal intializer parameter
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The model size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    # Do input normalization as was done in the the Annotated Transformer
    # page: http://nlp.seas.harvard.edu/2018/04/03/attention.html and in BERT.
    with tf.variable_scope("input_norm"):
        prev_output = tf.contrib.layers.layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1)
        prev_output = utils.dropout(prev_output, dropout_prob=dropout_prob)

    # Repeat to build multiple layers
    embedding_size = input_tensor.shape.as_list()[2]
    for layer_idx in range(num_encoder_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            with tf.variable_scope("self_attn"):
                # Attention, ignore attention probabilities for now
                attn_output, _ = multiheaded_attention(
                    prev_output, prev_output, prev_output, attention_mask,
                    hidden_size=hidden_size,
                    head_count=num_attention_heads,
                    dropout_prob=dropout_prob,
                    initializer_range=initializer_range
                )
                # Add and normalizer
                prev_output += utils.dropout(attn_output, dropout_prob=dropout_prob)
                prev_output = tf.contrib.layers.layer_norm(
                    inputs=prev_output, begin_norm_axis=-1, begin_params_axis=-1)

            with tf.variable_scope("feed_forward"):
                # The feedforward sublayer
                intermediate_output = tf.layers.dense(
                    prev_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    name="dense_in",
                    kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
                )
                intermediate_output = tf.layers.dense(
                    intermediate_output,
                    embedding_size,
                    activation=intermediate_act_fn,
                    name="dense_out",
                    kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
                )

                # Add and normalize
                prev_output += utils.dropout(intermediate_output, dropout_prob=dropout_prob)
                prev_output = tf.contrib.layers.layer_norm(
                    inputs=prev_output, begin_norm_axis=-1, begin_params_axis=-1)

    return prev_output

def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = tf.shape(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = tf.shape(to_mask)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask
