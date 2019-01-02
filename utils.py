"""Some utility functions and classes"""
import six
import tensorflow as tf

def dropout(input_x, dropout_prob=None):
    """The model method implementation"""
    if dropout_prob is None or dropout_prob == 0.0:
        return input_x
    return tf.nn.dropout(input_x, 1.0 - dropout_prob)

def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" % \
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf

def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)

def word_embedding_lookup(input_ids,
                          vocab_size,
                          embedding_size=128,
                          initializer_range=0.02,
                          use_one_hot_embeddings=False,
                          word_embedding_name="word_embeddings"):
    """Looks up words embeddings for id tensor.

    Args:
        input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
            ids.
        vocab_size: int. Size of the embedding vocabulary.
        embedding_size: int. Width of the word embeddings.
        initializer_range: float. Embedding initialization range.
        use_one_hot_embeddings: bool. Using one hot encoding in TPU is much faster,
            while not using it is faster in GPU.
        word_embedding_name: string. Name of the embedding table.

    Returns:
        float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length].
    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    if use_one_hot_embeddings:
        input_shape = input_ids.shape.as_list()
        input_shape[0] = tf.shape(input_ids)[0] # batch_size is dynamic
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
        output = tf.reshape(
            output, input_shape + [embedding_size])
    else:
        output = tf.nn.embedding_lookup(embedding_table, input_ids)
    return (output, embedding_table)

def token_type_embedding_lookup(token_type_ids,
                                vocab_size=16,
                                embedding_size=128,
                                initializer_range=0.02,
                                embedding_name="token_type_embeddings"):
    """Token type embedding lookup

    Args:
        token_type_ids: Int32 tensor of shape [batch_size, seq_length] containing
                        token type ids
        vocab_size: int, size of embedding vocabulary. Usually small.
        embedding_size: int. Width of token type embeddings
        initializer_range: float. Truncated normalized initialization range.
        embedding_name: string. Name of the embedding table

    Returns:
        float tensor of shape [batch_size, seq_length, embedding_size].
    """
    input_shape = tf.shape(token_type_ids)
    token_type_table = tf.get_variable(
        name=embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(
        token_type_embeddings, [input_shape[0], input_shape[1], embedding_size])
    return token_type_embeddings, token_type_table

def position_embedding_lookup(seq_length,
                              max_position_embeddings=512,
                              embedding_size=128,
                              initializer_range=0.02,
                              embedding_name="position_embeddings"):
    """Position embedding lookup. Instead of passing ids in typical embedding lookups,
    here we pass the seq_length, in essence, to lookup for ids of [0, 1, ..., seq_length-1]

    Args:
        seq_length: int32. sequence length of the batch.
        max_position_embeddings: int. The maximum number of position embeddings.
        embedding_size: int. The width of position embedding.
        initializer_range: float. The range for truncated normal initialization.
        embedding_name: string. Name of the embedding table

    Returns:
        float tensor of shape [batch_size, seq_length, embedding_size].
    """
    position_embedding_table = tf.get_variable(
        name=embedding_name,
        shape=[max_position_embeddings, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
        position_embeddings = tf.slice(
            position_embedding_table, [0, 0], [seq_length, -1])
        position_embeddings = tf.expand_dims(position_embeddings, axis=[0])
    return position_embeddings, position_embedding_table
