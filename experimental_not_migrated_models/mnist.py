from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import common
import image_dataset

import os
import time
import pprint
import numpy as np
import tensorflow as tf

# GENERAL
IMAGE_SIZE = 28
MODEL_DIR = "../models/mnist"

EVAL_EVERY_N_TRAIN_STEPS = 1000

tf.logging.set_verbosity(tf.logging.INFO)

# TRAINING
USE_STATIC_LEARNING_RATE = False
LEARNING_RATE_INITIAL = 0.015
LEARNING_RATE_DECAY_RATE = 0.96
LEARNING_RATE_DECAY_STEPS = 1000

DROPOUT_RATE = 0.6
TRAINING_EPOCHS = 1
TRAINING_STEPS = 2000
TRAINING_BATCH_SIZE = 500

# EVALUATION
EVAL_BATCH_SIZE = 100
VALIDATION_DATA_SIZE = 10000
EVAL_STEPS = VALIDATION_DATA_SIZE // EVAL_BATCH_SIZE

# MODEL CONFIG
model_configs = {
    "test1": {
        "skip": False,
        "conv1": {
            "filters": 64,
            "kernel_size": [5, 5],
            "strides": (1, 1)
        },
        "pool1": {
            "pool_size": [3, 3],
            "strides": 2
        },
        "conv2": {
            "filters": 172,
            "kernel_size": [3, 3],
            "strides": (1, 1)
        },
        # "conv3": {
        #     "filters": 128,
        #     "kernel_size": [3, 3],
        #     "strides": (1, 1)
        # },
        "pool2": {
            "pool_size": [3, 3],
            "strides": 2
        },
        "dense1": {
            "units": 2048
        },
        "dense2": {
            "units": 512
        }
    }
}

model_details = {
    "params": {}
}


def get_learning_rate():
    if USE_STATIC_LEARNING_RATE:
        learning_rate = LEARNING_RATE_INITIAL
    else:
        learning_rate = tf.train.exponential_decay(
            learning_rate=LEARNING_RATE_INITIAL,
            global_step=tf.train.get_global_step(),
            decay_steps=LEARNING_RATE_DECAY_STEPS,
            decay_rate=LEARNING_RATE_DECAY_RATE,
            name="learning_rate"
        )

    tf.summary.scalar("learning_rate", learning_rate)

    return learning_rate

# norm1 = tf.nn.local_response_normalization(pool1, 4, alpha=0.00011, beta=0.75, name="norm1")


def model_fn(features, labels, mode, params, config):
    """Model function for CNN."""
    model_details["params"] = params.copy()
    model_details["model_dir"] = config.model_dir

    # Input Layer
    with tf.name_scope("input_layer"):
        input_layer = tf.reshape(features["x"], [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    print(input_layer.shape, features["x"].shape)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=params["conv1"]["filters"],
        kernel_size=params["conv1"]["kernel_size"],
        strides=params["conv1"]["strides"],
        padding="same",
        activation=tf.nn.relu,
        name="conv1"
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=params["pool1"]["pool_size"],
        strides=params["pool1"]["strides"],
        name="pool1"
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=params["conv2"]["filters"],
        kernel_size=params["conv2"]["kernel_size"],
        strides=params["conv2"]["strides"],
        padding="same",
        activation=tf.nn.relu,
        name="conv2"
    )
    # conv3 = tf.layers.conv2d(
    #     inputs=pool1,
    #     filters=params["conv3"]["filters"],
    #     kernel_size=params["conv3"]["kernel_size"],
    #     strides=params["conv3"]["strides"],
    #     padding="same",
    #     activation=tf.nn.relu,
    #     name="conv3"
    # )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=params["pool2"]["pool_size"],
        strides=params["pool2"]["strides"],
        name="pool2"
    )

    # common.build_layer_summaries("conv1")
    # common.build_layer_summaries("conv2")

    pool2_flat = tf.reshape(pool2, [-1, np.multiply.reduce(pool2.shape[1:])], name="pool2_reshape")

    # Dense Layer
    dense1 = tf.layers.dense(
            inputs=pool2_flat, units=params["dense1"]["units"], activation=tf.nn.relu, name="dense1")

    dense2 = tf.layers.dense(
            inputs=dense1, units=params["dense2"]["units"], activation=tf.nn.relu, name="dense2")

    # GET MODEL DETAILS
    model_details["params"]["conv1"]["shape"] = conv1.shape.as_list()
    model_details["params"]["conv2"]["shape"] = conv2.shape.as_list()
    model_details["params"]["pool2_flat"] = {"shape": pool2_flat.shape.as_list()}
    model_details["params"]["dense1"]["shape"] = dense1.shape.as_list()
    model_details["params"]["dense2"]["shape"] = dense2.shape.as_list()

    with tf.name_scope("dropout"):
        dropout = tf.layers.dropout(
            inputs=dense2, rate=DROPOUT_RATE, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10, name="logits")

    argmax = tf.argmax(input=logits, axis=1, name="predictions")
    softmax = tf.nn.softmax(logits, name="softmax_tensor")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": argmax,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": softmax
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope="calc_loss")
    tf.summary.scalar("cross_entropy", loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = get_learning_rate()

        summary_saver = tf.train.SummarySaverHook(
            save_steps=50, output_dir=config.model_dir + "/train", summary_op=tf.summary.merge_all())

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate, name="gradient_descent_optimizer")
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step(), name="minimize_loss")

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_saver])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    pp = pprint.PrettyPrinter(indent=2, compact=True)

    # Load training and eval data
    (train_x, train_y), (test_x, test_y) = common.load_original_mnist()


    print(train_x.shape, train_x.dtype, train_y.shape, train_y.dtype)
    print(test_x.shape, test_x.dtype, test_y.shape, test_y.dtype)

    def train_model(classifier, log_stats=True):
        start_time = time.time()

        # Train the model
        # profiler_hook = tf.train.ProfilerHook(save_steps=50, output_dir=MODEL_DIR + '/train')

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_x},
            y=train_y,
            batch_size=TRAINING_BATCH_SIZE,
            num_epochs=None,
            shuffle=True
        )
        classifier.train(
            input_fn=train_input_fn,
            steps=TRAINING_STEPS,
            # hooks=[profiler_hook]
        )
        duration = round(time.time() - start_time, 3)

        if log_stats:
            print("Training duration: " + common.duration_to_string(duration))

        return duration

    def eval_model(classifier, log_stats=True):
        start_time = time.time()

        tensors_to_log = {
            # "probabilities": "softmax_tensor",
            "pred": "diff"
        }
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_x},
            y=test_y,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False
        )
        result = classifier.evaluate(
            input_fn=eval_input_fn,
            steps=EVAL_STEPS,
            # hooks=[logging_hook]
        )
        duration = round(time.time() - start_time, 3)

        if log_stats:
            print("Training duration: " + common.duration_to_string(duration))
            print("Eval result:", result)

        return result, duration

    model_stats_map = {}
    for conf_name, config in model_configs.items():

        # if config["skip"]:
        #     continue

        print("RUN CONFIG: %s" % conf_name)
        model_dir = os.path.join(MODEL_DIR, conf_name)

        # common.clean_dir(model_dir)

        mnist_classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            params=config
        )

        eval_results = []
        total_train_duration = 0
        total_eval_duration = 0
        for _ in range(TRAINING_EPOCHS):
            # train_duration = train_model(mnist_classifier)
            # total_train_duration += train_duration

            eval_result, eval_duration = eval_model(mnist_classifier)
            eval_results.append(eval_result)
            total_eval_duration += eval_duration

        final_result = common.get_final_eval_result(eval_results)

        print("Eval results:")
        pp.pprint(eval_results)
        model_stats_map[conf_name] = {
            "model_details": model_details,
            "final_result": final_result,
            "total_train_duration": common.duration_to_string(total_train_duration),
            "total_eval_duration": common.duration_to_string(total_eval_duration),
        }
        common.save_pickle(
            model_stats_map[conf_name],
            os.path.join(model_details["model_dir"], "last_result.pkl")
        )
        common.save_json(
            model_stats_map[conf_name],
            os.path.join(model_details["model_dir"], "last_result.json")
        )

        print("Total training duration: " + common.duration_to_string(total_train_duration))
        print("Total eval duration: " + common.duration_to_string(total_eval_duration))

    print("Models results:")
    pp.pprint(model_stats_map)


if __name__ == "__main__":
    tf.app.run(main=main)
