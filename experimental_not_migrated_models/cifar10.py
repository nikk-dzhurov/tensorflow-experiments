from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import common
from image import LabeledImage
from image_dataset import ImageDataset

import os
import time
import pprint
import numpy as np
import tensorflow as tf

# GENERAL
RANDOM_SEED = 55355
IMAGE_SIZE = 32
MODEL_DIR = "../models/cifar10"

EVAL_EVERY_N_TRAIN_STEPS = 1000

tf.logging.set_verbosity(tf.logging.INFO)

# TRAINING
USE_STATIC_LEARNING_RATE = False
LEARNING_RATE_INITIAL = 0.05
LEARNING_RATE_DECAY_RATE = 0.96
LEARNING_RATE_DECAY_STEPS = 10000

DROPOUT_RATE = 0.6
TRAINING_EPOCHS = 1
TRAINING_STEPS = 2000
TRAINING_BATCH_SIZE = 128

# EVALUATION
EVAL_BATCH_SIZE = 500
VALIDATION_DATA_SIZE = 10000
EVAL_STEPS = VALIDATION_DATA_SIZE // EVAL_BATCH_SIZE

# MODEL CONFIG
model_configs = {
    "test3": {
        "skip": False,
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


def model_fn(features, labels, mode, params, config):
    """Model function for CNN."""
    model_details["model_dir"] = config.model_dir

    # Input Layer
    with tf.name_scope("input_layer"):
        input_layer = features["x"]
        if mode == tf.estimator.ModeKeys.TRAIN and False:
            def fn(img):
                dist = tf.random_crop(img, [24, 24, 3])
                dist = tf.image.random_contrast(dist, lower=0.2, upper=1.8)
                # dist = tf.image.random_hue(dist, max_delta=0.2)
                dist = tf.image.random_flip_left_right(dist)
                # dist = tf.image.random_flip_up_down(dist)
                return dist

            input_layer = tf.map_fn(
                fn, input_layer, parallel_iterations=100)
            input_layer = tf.image.resize_image_with_crop_or_pad(input_layer, 32, 32)

    conv3x3 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[3, 3],
        strides=1,
        padding="valid",
        activation=tf.nn.relu,
        name="conv3x3"
    )
    # conv5x5 = tf.layers.conv2d(
    #     inputs=input_layer,
    #     filters=48,
    #     kernel_size=[5, 5],
    #     strides=1,
    #     padding="same",
    #     activation=tf.nn.relu,
    #     name="conv5x5"
    # )
    #
    # mixed = tf.concat([conv3x3, conv5x5], axis=3, name="mixed_layer")

    pool1 = tf.layers.max_pooling2d(
        inputs=conv3x3,
        pool_size=[2, 2],
        strides=2,
        name="pool1"
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        strides=1,
        padding="same",
        activation=tf.nn.relu,
        name="conv2"
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name="pool2"
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu,
        name="conv3"
    )
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=2,
        name="pool3"
    )

    pool_flat = tf.reshape(pool3, [-1, pool3.shape[1]*pool3.shape[2]*pool3.shape[3]], name="pool2_reshape")

    # Dense Layer
    dense1 = tf.layers.dense(
            inputs=pool_flat, units=512, activation=tf.nn.relu, name="dense1")

    dense2 = tf.layers.dense(
            inputs=dense1, units=384, activation=tf.nn.relu, name="dense2")

    dense3 = tf.layers.dense(
        inputs=dense1, units=256, activation=tf.nn.relu, name="dense3")

    # GET MODEL DETAILS
    model_details["params"]["conv3x3"] = {"shape": conv3x3.shape.as_list()}
    # model_details["params"]["conv5x5"] = {"shape": conv5x5.shape.as_list()}
    # model_details["params"]["mixed"] = {"shape": mixed.shape.as_list()}
    model_details["params"]["conv2"] = {"shape": conv2.shape.as_list()}
    model_details["params"]["conv3"] = {"shape": conv3.shape.as_list()}
    model_details["params"]["pool1"] = {"shape": pool1.shape.as_list()}
    model_details["params"]["pool2"] = {"shape": pool2.shape.as_list()}
    model_details["params"]["pool3"] = {"shape": pool3.shape.as_list()}
    model_details["params"]["pool_flat"] = {"shape": pool_flat.shape.as_list()}
    model_details["params"]["dense1"] = {"shape": dense1.shape.as_list()}
    model_details["params"]["dense2"] = {"shape": dense2.shape.as_list()}
    model_details["params"]["dense3"] = {"shape": dense3.shape.as_list()}

    with tf.name_scope("dropout"):
        dropout = tf.layers.dropout(
            inputs=dense3, rate=DROPOUT_RATE, training=mode == tf.estimator.ModeKeys.TRAIN)

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
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="adam_optimizer")
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step(), name="minimize_loss")

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_saver])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def load_cifar10():
    train_ds = ImageDataset.load_from_pickles([
        "/datasets/cifar10/original_train.pkl",
        "/datasets/cifar10/mirror_train.pkl",
        "/datasets/cifar10/rand_distorted_train.pkl",
    ])
    test_ds = ImageDataset.load_from_pickles([
        "/datasets/cifar10/original_test.pkl",
    ])

    offset = 1024
    img0 = LabeledImage.load_from_dataset_tuple((train_ds.x, train_ds.y), 0 + offset)
    img1 = LabeledImage.load_from_dataset_tuple((train_ds.x, train_ds.y), 50000 + offset)
    img2 = LabeledImage.load_from_dataset_tuple((train_ds.x, train_ds.y), 100000 + offset)

    mixed_img = np.concatenate([img0.image, img1.image, img2.image], axis=1)
    LabeledImage(mixed_img, "mixed_" + img0.name) \
        .save_image()

    return (train_ds.x, train_ds.y), (test_ds.x, test_ds.y)

def main(unused_argv):
    pp = pprint.PrettyPrinter(indent=2, compact=True)

    # Load training and eval data
    (train_x, train_y), (test_x, test_y) = load_cifar10()

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
            print("Eval duration: " + common.duration_to_string(duration))
            print("Eval result:", result)

        return result, duration

    model_stats_map = {}
    for params_name, params in model_configs.items():

        # if config["skip"]:
        #     continue

        print("RUN PARAMS: %s" % params_name)
        model_dir = os.path.join(MODEL_DIR, params_name)

        # common.clean_dir(model_dir)

        # Reduce GPU memory usage per process
        sess_config = tf.ConfigProto()
        # sess_config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3))

        classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            params=params,
            config=tf.estimator.RunConfig(session_config=sess_config)
        )

        eval_results = []
        total_train_duration = 0
        total_eval_duration = 0
        for i in range(TRAINING_EPOCHS):
            train_duration = train_model(classifier)
            total_train_duration += train_duration

            eval_result, eval_duration = eval_model(classifier)
            eval_results.append(eval_result)
            total_eval_duration += eval_duration

            print("Epoch %d of %d completed" % (i, TRAINING_EPOCHS))

        final_result = common.get_final_eval_result(eval_results)

        print("Eval results:")
        pp.pprint(eval_results)
        model_stats_map[params_name] = {
            "model_details": model_details,
            "final_result": final_result,
            "total_train_duration": common.duration_to_string(total_train_duration),
            "total_eval_duration": common.duration_to_string(total_eval_duration),
        }
        common.save_pickle(
            model_stats_map[params_name],
            os.path.join(model_details["model_dir"], "last_result.pkl")
        )
        common.save_json(
            model_stats_map[params_name],
            os.path.join(model_details["model_dir"], "last_result.json")
        )

        print("Total training duration: " + common.duration_to_string(total_train_duration))
        print("Total eval duration: " + common.duration_to_string(total_eval_duration))

    print("Models results:")
    pp.pprint(model_stats_map)


if __name__ == "__main__":
    tf.app.run(main=main)
