"""
A transformer-based NLP encoder model.

It includes embedding layers for learned word embedding, token type
embedding and position embedding.

This model is currently based on and initialize from BERT's model
structure so that we can avoid expensive pretraining for now. But
the model codes are refactored to avoid reliance on BERT except
the variable names.
"""
import re
import collections
import tensorflow as tf

import utils
import transformer

def nlp_encoder(
        vocab_size,
        input_ids,
        input_mask=None,
        token_type_ids=None,
        token_type_vocab_size=2,
        embedding_size=768,
        attn_hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        intermediate_act_fn="gelu",
        num_encoder_layers=12,
        dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        use_one_hot_embeddings=False,
        scope=None
    ):
    """
    Transformer-based NLP encoder with embeddings.

    Args:
        vocab_size: the size of the word embedding vocabulary.
        input_ids: the ids of the input sequence. It should be
                   int tensor of shape [batch_size, seq_length].
        input_mask: Int tensor of shape [batch_size, seq_length]
                    with value 1 for valid input tokens, and
                    for invalid input token positions. If this is
                    None, we assume all input tokens are valid.
        token_type_ids: token type ids. Int tensor of shape
                        [batch_size, seq_length]. If this is None,
                        we initialize token type ids to zero.
        token_type_vocab_size: the vocabulary size for token
                               types.
        embedding_size: the size of embedding.
        attn_hidden_size: the hidden size of the attention layer.
        num_attention_heads: the number of attention heads.
        intermediate_size: the intermediate size for the feed forward
                           network of the transformer.
        intermediate_act_fn: the activation function for the
                             feed forward sub layers.
        num_encoder_layers: the number of encoding layers.
        dropout_prob: the dropout probabilities for both the attention
                      sub layers and feed forward sub layers.
        max_position_embeddings: the maximum position embedding.
        initializer_range: the range of the truncated normal
                           initialization for weights.
        use_one_hot_embeddings: if to use one hot embedding for
                                word embedding lookup - faster on TPU.
        scope: variable scope. Optional.
    """
    input_shape = tf.shape(input_ids)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    if input_mask is None:
        input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
    if token_type_ids is None:
        token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
        with tf.variable_scope("embeddings"):
            # Perform embedding lookup on the word ids.
            embedding_output, embedding_table = utils.word_embedding_lookup(
                input_ids=input_ids,
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                initializer_range=initializer_range,
                use_one_hot_embeddings=use_one_hot_embeddings)

            # Add positional embeddings and token type embeddings
            # layer normalize and dropout are taken care of in the transformer model.
            token_type_embedding_output, token_type_embedding_table = \
                utils.token_type_embedding_lookup(
                    token_type_ids,
                    vocab_size=token_type_vocab_size,
                    embedding_size=embedding_size,
                    initializer_range=initializer_range)
            position_embedding_output, position_embedding_table = \
                utils.position_embedding_lookup(
                    seq_length,
                    embedding_size=embedding_size,
                    initializer_range=initializer_range,
                    max_position_embeddings=max_position_embeddings)
            embedding_output += token_type_embedding_output
            embedding_output += position_embedding_output

        with tf.variable_scope("encoder"):
            # Run the stacked transformer.
            # `sequence_output` shape = [batch_size, seq_length, hidden_size].
            attention_mask = transformer.create_attention_mask_from_input_mask(
                input_ids, input_mask)
            sequence_output = transformer.transformer_encoder(
                input_tensor=embedding_output,
                attention_mask=attention_mask,
                hidden_size=attn_hidden_size,
                num_encoder_layers=num_encoder_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                intermediate_act_fn=utils.get_activation(intermediate_act_fn),
                dropout_prob=dropout_prob,
                initializer_range=initializer_range)
    return sequence_output, embedding_table, token_type_embedding_table, position_embedding_table

GENERAL_NAME_MAP = {
    'bert/embeddings/LayerNorm/beta' : 'bert/encoder/input_norm/LayerNorm/beta',
    'bert/embeddings/LayerNorm/gamma' : 'bert/encoder/input_norm/LayerNorm/gamma'
}
# layer_name_map has prefix string as "bert/encoder/layer_0/"
LAYER_NAME_MAP = {
    'attention/output/LayerNorm/beta' :     'self_attn/LayerNorm/beta',
    'attention/output/LayerNorm/gamma' :    'self_attn/LayerNorm/gamma',
    'attention/output/dense/bias' :         'self_attn/projection/bias',
    'attention/output/dense/kernel' :       'self_attn/projection/kernel',
    'attention/self/key/bias' :             'self_attn/key/bias',
    'attention/self/key/kernel' :           'self_attn/key/kernel',
    'attention/self/query/bias' :           'self_attn/query/bias',
    'attention/self/query/kernel' :         'self_attn/query/kernel',
    'attention/self/value/bias' :           'self_attn/value/bias',
    'attention/self/value/kernel' :         'self_attn/value/kernel',
    'intermediate/dense/bias' :             'feed_forward/dense_in/bias',
    'intermediate/dense/kernel' :           'feed_forward/dense_in/kernel',
    'output/LayerNorm/beta' :               'feed_forward/LayerNorm/beta',
    'output/LayerNorm/gamma' :              'feed_forward/LayerNorm/gamma',
    'output/dense/bias' :                   'feed_forward/dense_out/bias',
    'output/dense/kernel' :                 'feed_forward/dense_out/kernel'
}

def map_bert_variable_name(bert_name):
    """We renamed some of the model variables, but we'd still
    like to be able to import Bert's pretrained model. So we
    need to map Bert's variable name in their model file to our
    names"""
    if bert_name in GENERAL_NAME_MAP.keys():
        return GENERAL_NAME_MAP[bert_name]
    else:
        name_match = re.match("^bert/encoder/layer_(\\d+)/(.*)$", bert_name)
        if name_match is not None:
            layer_idx = name_match.group(1)
            remaining = name_match.group(2)
            if remaining in LAYER_NAME_MAP.keys():
                return "bert/encoder/layer_" + layer_idx + "/" + LAYER_NAME_MAP[remaining]
        return bert_name

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m_var = re.match("^(.*):\\d+$", name)
        if m_var is not None:
            name = m_var.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x_var in init_vars:
        (bert_name, var) = (x_var[0], x_var[1])
        if bert_name not in name_to_variable:
            name = map_bert_variable_name(bert_name)
        else:
            name = bert_name
        if name not in name_to_variable:
            continue
        assignment_map[bert_name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)
