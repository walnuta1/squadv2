"""Trains models for SQuAD v2 and runs predictions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import math
import json
import six

import tensorflow as tf

import optimization
import tokenization
import feature_store
import model_encoder
import model_decoder

# The default flags values set here correspond to
# BERT large model parameters in bert_config.json.

tf.flags.DEFINE_integer(
    "vocab_size", 30522,
    "Word embedding vocabulary size"
)

tf.flags.DEFINE_integer(
    "token_type_vocab_size", 2,
    "The token type embedding vocabulary size"
)

tf.flags.DEFINE_integer(
    "embedding_size", 1024,
    "The word embedding size/dimension"
)

tf.flags.DEFINE_integer(
    "attn_hidden_size", 1024,
    "The hidden size for attention layers"
)

tf.flags.DEFINE_integer(
    "num_attention_heads", 16,
    "The number of attention heads for attention layers"
)

tf.flags.DEFINE_integer(
    "intermediate_size", 4096,
    "The intermediate size for the feed forward networks"
)

tf.flags.DEFINE_string(
    "intermediate_act_fn", "gelu",
    "The activiation function for the intermediate layer of the feed forward networks"
)

tf.flags.DEFINE_integer(
    "num_encoder_layers", 24,
    "The number of encoder layers in the encoder module"
)

tf.flags.DEFINE_float(
    "dropout_prob", 0.1,
    "The default dropout probability for the transfomer model"
)

tf.flags.DEFINE_integer(
    "max_position_embeddings", 512,
    "The maximum position value for position embedding"
)

tf.flags.DEFINE_float(
    "initializer_range", 0.02,
    "The weight initialization range for truncated normal initialization"
)

tf.flags.DEFINE_float(
    "learning_rate", 5e-5,
    "The initial learning rate for Adam."
)

tf.flags.DEFINE_float(
    "num_train_epochs", 3.0,
    "Total number of training epochs to perform."
)

tf.flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training."
)

tf.flags.DEFINE_boolean(
    "use_tpu", False,
    "If to use TPU for training & inference"
)

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

tf.flags.DEFINE_integer("save_checkpoints_steps", 1000,
                        "How often to save the model checkpoint.")

tf.flags.DEFINE_integer("iterations_per_loop", 1000,
                        "How many steps to make in each estimator call.")

tf.flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

tf.flags.DEFINE_float(
    "answerability_weight", 1.0,
    "The relative weight of the answerability loss vs positional losses"
)

tf.flags.DEFINE_float(
    "prediction_error_weight", 10.0,
    "The relative positional prediction error weight vs normal positional losses"
)

tf.flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on."
)

tf.flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

tf.flags.DEFINE_bool("do_train", False, "Whether to run training.")

tf.flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

tf.flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

tf.flags.DEFINE_integer("predict_batch_size", 8,
                        "Total batch size for predictions.")

tf.flags.DEFINE_string("train_file", None,
                       "SQuAD json for training. E.g., train-v1.1.json")

tf.flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

tf.flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

tf.flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

tf.flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

tf.flags.DEFINE_float(
    "na_prob_threshold", 0.5,
    "If na_prob is greater than the threshold predict null/not-answerable.")

tf.flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

tf.flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

tf.flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

tf.flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

FLAGS = tf.flags.FLAGS

def model_function(features, labels, mode, params):    # pylint: disable=unused-argument
    """Model function for TPUEstimator"""

    # Dump all feature names and shapes
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("    name = %s, shape = %s" % (name, features[name].shape))

    # Extract and referenc features by name
    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    # Adjust dropout_prob for different modes
    dropout_prob = FLAGS.dropout_prob
    if mode != tf.estimator.ModeKeys.TRAIN:
        dropout_prob = 0.0

    # Construct a full model from encoder and decoder modules
    sequence_output, _, _, _ = model_encoder.nlp_encoder(
        FLAGS.vocab_size,
        input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        token_type_vocab_size=FLAGS.token_type_vocab_size,
        embedding_size=FLAGS.embedding_size,
        attn_hidden_size=FLAGS.attn_hidden_size,
        num_attention_heads=FLAGS.num_attention_heads,
        intermediate_size=FLAGS.intermediate_size,
        intermediate_act_fn=FLAGS.intermediate_act_fn,
        num_encoder_layers=FLAGS.num_encoder_layers,
        dropout_prob=dropout_prob,
        max_position_embeddings=FLAGS.max_position_embeddings,
        initializer_range=FLAGS.initializer_range,
        use_one_hot_embeddings=FLAGS.use_tpu
    )
    answerable_logits, start_pos_logits, end_pos_logits = model_decoder.squad_v2_decoder(
        sequence_output,
        input_mask,
        segment_ids,
        embedding_size=FLAGS.embedding_size,
        dropout_prob=dropout_prob,
        initializer_range=FLAGS.initializer_range
    )

    # Handle model scaffolding in case we are given an initial checkpoint
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if FLAGS.init_checkpoint:
        assignment_map, initialized_variable_names = \
            model_encoder.get_assignment_map_from_checkpoint(
                tvars, FLAGS.init_checkpoint
            )
        if FLAGS.use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(
                    FLAGS.init_checkpoint, assignment_map
                )
                return tf.train.Scaffold()
            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(
                FLAGS.init_checkpoint, assignment_map
            )

    # Dump information about variable initialization
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        else:
            init_string = ""
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    # Construct output_spec for training and prediction
    # respetively
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        # In training mode, compute loss & construct optimizer
        seq_length = tf.shape(input_ids)[1]

        is_impossible = features["is_impossible"]
        start_position = features["start_positions"]
        end_position = features["end_positions"]

        def compute_loss(logits, logit_labels, logit_depth):
            one_hot_positions = tf.one_hot(
                logit_labels, depth=logit_depth, dtype=tf.float32)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            loss = -tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
            return loss

        answerable_loss = compute_loss(answerable_logits, is_impossible, 2)
        start_pos_loss = compute_loss(
            start_pos_logits, start_position, seq_length)
        end_pos_loss = compute_loss(end_pos_logits, end_position, seq_length)

        total_loss = math.log(FLAGS.max_seq_length / 2.0) \
                    * FLAGS.answerability_weight * answerable_loss + \
                    (start_pos_loss + end_pos_loss) * (1.0 - tf.cast(is_impossible, tf.float32))
        final_total_loss = tf.reduce_mean(total_loss)

        train_op = optimization.create_optimizer(
            final_total_loss,
            FLAGS.learning_rate,
            params["num_train_steps"],
            params["num_warmup_steps"],
            FLAGS.use_tpu)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=final_total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        # In prediction mode
        predictions = {
            "unique_ids": unique_ids,
            "answerable_logits": answerable_logits,
            "start_logits": start_pos_logits,
            "end_logits": end_pos_logits,
        }
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
        raise ValueError(
            "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """
    Creates an `input_fn` closure to be passed to TPUEstimator.
    The input_fn assumes a TFRecord file format, and thus preprocessing
    is needed to convert training/prediction data file into TFRecord
    file format.
    """

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            var_t = example[name]
            if var_t.dtype == tf.int64:
                var_t = tf.cast(var_t, tf.int32)
            example[name] = var_t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        data = tf.data.TFRecordDataset(input_file)
        if is_training:
            data = data.repeat()
            data = data.shuffle(buffer_size=100)

        data = data.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return data

    return input_fn

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i, item in enumerate(index_and_score):
        if i >= n_best_size:
            break
        best_indexes.append(item[0])
    return best_indexes

def _get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #     pred_text = steve smith
    #     orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, text_char) in enumerate(text):
            if text_char == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(text_char)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        exp = math.exp(score - max_score)
        exp_scores.append(exp)
        total_sum += exp

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def _get_na_prob(logits):
    probs = _compute_softmax(logits)
    return probs[1]

def write_predictions(eval_examples, eval_features, all_results,
                      max_answer_length, do_lower_case,
                      output_prediction_file,
                      output_nbest_file,
                      na_probs_file):
    """
    Given evaluation examples, converted features and prediction results,
    Write prediction outputs in the required JSON file format
    """
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in eval_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    na_probs_json = collections.OrderedDict()

    _PrelimPrediction = collections.namedtuple(    # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", \
         "na_prob", "start_logit", "end_logit"])

    for (example_index, example) in enumerate(eval_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, FLAGS.n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, FLAGS.n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            na_prob=_get_na_prob(result.answerable_logits),
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (1.0 - x.na_prob, x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(    # pylint: disable=invalid-name
            "NbestPrediction", \
            ["text", "na_prob", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= FLAGS.n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:    # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = _get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                pred.na_prob = 1.0
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    na_prob=pred.na_prob,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", na_prob=1.0,
                                 start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        #if nbest[0].na_prob >= FLAGS.na_prob_threshold:
            #all_predictions[example.qas_id] = ""
        #else:
        all_predictions[example.qas_id] = nbest[0].text
        all_nbest_json[example.qas_id] = nbest
        na_probs_json[example.qas_id] = nbest[0].na_prob

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    with tf.gfile.GFile(na_probs_file, "w") as writer:
        writer.write(json.dumps(na_probs_json, indent=4) + "\n")

def main(_):
    """The main execution function"""

    tf.logging.set_verbosity(tf.logging.INFO)

    # Parameter validations
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")
    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")
    if FLAGS.use_tpu and FLAGS.tpu_name is None:
        raise ValueError(
            "`tpu_name` must be specified if `use_tpu` is True"
        )

    RawResult = collections.namedtuple(
        "RawResult", ["unique_id", "answerable_logits", "start_logits", "end_logits"])

    # Create output directory, with gfile to support GCP storage buckets
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # Convert training file format to TFRecord
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_record_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not tf.gfile.Exists(train_record_file):
            tf.logging.info("***** Converting training examples *****")
            examples, features = feature_store.convert_squad_data_file_to_tf_record_file(
                FLAGS.train_file, True, train_record_file,
                tokenizer=tokenizer,
                max_seq_length=FLAGS.max_seq_length,
                doc_stride=FLAGS.doc_stride,
                max_query_length=FLAGS.max_query_length,
                is_version_2_with_negative=True
            )
            num_examples = len(examples)
            num_features = len(features)
            tf.logging.info("    Num orig examples = %d", num_examples)
            tf.logging.info("    Num split examples = %d", num_features)
        else:
            num_features = feature_store.get_tf_record_count(train_record_file)
        num_train_steps = int(
            num_features / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Prepare the run_config object
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # Construct TPUEstimator and run training
    estimator_params = {
        "num_warmup_steps": num_warmup_steps,
        "num_train_steps": num_train_steps
    }
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_function,
        config=run_config,
        params=estimator_params,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("    Num split examples = %d", num_features)
        tf.logging.info("    Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("    Num steps = %d", num_train_steps)
        del train_examples

        train_input_fn = input_fn_builder(
            input_file=train_record_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        eval_record_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        tf.logging.info("***** Converting eval examples *****")
        eval_examples, eval_features = feature_store.convert_squad_data_file_to_tf_record_file(
            FLAGS.predict_file, False, eval_record_file,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_version_2_with_negative=True
        )
        num_examples = len(eval_examples)
        num_features = len(eval_features)
        tf.logging.info("    Num orig examples = %d", num_examples)
        tf.logging.info("    Num split examples = %d", num_features)

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("    Num orig examples = %d", len(eval_examples))
        tf.logging.info("    Num split examples = %d", len(eval_features))
        tf.logging.info("    Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = input_fn_builder(
            input_file=eval_record_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            answerable_logits = [float(x) for x in result["answerable_logits"].flat]
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    answerable_logits=answerable_logits,
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
        output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
        na_probs_file = os.path.join(FLAGS.output_dir, "na_probs.json")

        write_predictions(eval_examples, eval_features, all_results,
                          FLAGS.max_answer_length,
                          FLAGS.do_lower_case,
                          output_prediction_file,
                          output_nbest_file,
                          na_probs_file)

if __name__ == "__main__":
    tf.flags.mark_flag_as_required("vocab_file")
    tf.flags.mark_flag_as_required("output_dir")
    tf.app.run()
