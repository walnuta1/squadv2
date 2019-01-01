"""
Replication of BERT's modeling, to double check the correctness of rewritten codes.
"""
import copy
import json
import re
import collections
import six
import tensorflow as tf

import utils
import transformer

class BertConfig(object):
    """Configuration for `BertModel`. Copied straight out of
    https://github.com/google-research/bert/blob/master/modeling.py"""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertModel(object):
    """The BertModel as in https://github.com/google-research/bert/blob/master/modeling.py

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """Constructor for BertModel.

        Args:
        config: `BertConfig` instance.
        is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
        input_ids: int32 Tensor of shape [batch_size, seq_length].
        input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
            it is much faster if this is True, on the CPU or GPU, it is faster if
            this is False.
        scope: (optional) variable scope. Defaults to "bert".

        Raises:
        ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

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
                self.embedding_output, self.embedding_table = utils.word_embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings
                # layer normalize and dropout are taken care of in the transformer model.
                self.token_type_embedding_output, self.token_type_embedding_table = \
                    utils.token_type_embedding_lookup(
                        token_type_ids,
                        vocab_size=config.type_vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range)
                self.position_embedding_output, self.position_embedding_table = \
                    utils.position_embedding_lookup(
                        seq_length,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings)
                self.embedding_output += self.token_type_embedding_output
                self.embedding_output += self.position_embedding_output

            with tf.variable_scope("encoder"):
                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.sequence_output = transformer.transformer_encoder(
                    input_tensor=self.embedding_output,
                    attention_mask=input_mask,
                    hidden_size=config.hidden_size,
                    num_encoder_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=utils.get_activation(config.hidden_act),
                    dropout_prob=config.hidden_dropout_prob,
                    initializer_range=config.initializer_range)

            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=tf.truncated_normal_initializer(
                        stddev=config.initializer_range))

    def get_pooled_output(self):
        """Gets poooled output"""
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
        to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
        to the output of the embedding layer, after summing the word
        embeddings with the positional embeddings and the token type embeddings,
        then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        """Gets the word embedding table"""
        return self.embedding_table

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
