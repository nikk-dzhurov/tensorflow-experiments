from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import common
from image import LabeledImage
import image_dataset as ds

import os
import time
import pprint
import numpy as np
import tensorflow as tf


# Set logging level
tf.logging.set_verbosity(tf.logging.INFO)


# GENERAL


# MODEL CONFIG
model_configs = {
    "test1": {
        "skip": False,
    }
}

model_details = {
    "params": {}
}


class Classifier(object):
    def __init__(
            self,
            model_fn,
            model_params,
            load_eval_ds_fn,
            load_train_ds_fn,
            clean_old_model_data=False):

        run_config = self._get_run_config_from_flags()

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=tf.app.flags.FLAGS.model_dir,
            params=model_params,
            config=run_config
        )

        self.load_eval_ds_fn = load_eval_ds_fn
        self.load_train_ds_fn = load_train_ds_fn

        self.eval_ds = None
        self.train_ds = None
        self.eval_results = []
        self.total_eval_duration = 0
        self.total_train_duration = 0

        if clean_old_model_data:
            common.clean_dir(tf.app.flags.FLAGS.model_dir)

    @staticmethod
    def _print_ds_details(self, ds, ds_name=""):
        print(f"{ds_name}_x:\n\t shape: {self.train_ds[0].shape}\n\t type: {self.train_ds[0].dtype}")
        print(f"{ds_name}_x:\n\t shape: {self.train_ds[1].shape}\n\t type: {self.train_ds[1].dtype}")

    @staticmethod
    def _get_run_config_from_flags(self):
        flags = tf.app.flags.FLAGS
        sess_config = tf.ConfigProto()

        if flags.ignore_gpu:
            sess_config = tf.ConfigProto(device_count={'GPU': 0})
        elif flags.per_process_gpu_memory_fraction is not None:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=flags.per_process_gpu_memory_fraction)
            sess_config = tf.ConfigProto(gpu_options=gpu_options)

        return tf.estimator.RunConfig(session_config=sess_config)

    def get_eval_ds(self):
        if self.eval_ds is None:
            self.eval_ds = self.load_eval_ds_fn()
            print("Train dataset loaded successfully!")
            self._print_ds_details(self.train_ds, "eval")

        return self.eval_ds

    def get_train_ds(self):
        if self.train_ds is None:
            self.train_ds = self.load_train_ds_fn()
            print("Train dataset loaded successfully!")
            self._print_ds_details(self.train_ds, "train")

        return self.train_ds

    def _print_results(self):
        if len(self.eval_results) > 0:
            final_result = common.get_final_eval_result(eval_results)


        # FINISH!
        model_stats_map = {
            "model_details": model_details,
            "final_result": final_result,
            "total_train_duration": common.duration_to_string(self.total_train_duration),
            "total_eval_duration": common.duration_to_string(self.total_eval_duration),
        }
        common.save_pickle(
            model_stats_map,
            os.path.join(model_details["model_dir"], "last_result.pkl")
        )
        common.save_json(
            model_stats_map[params_name],
            os.path.join(model_details["model_dir"], "last_result.json")
        )

        print("Total training duration: " + common.duration_to_string(total_train_duration))
        print("Total eval duration: " + common.duration_to_string(total_eval_duration))

    def train(self, steps, epochs=1, eval_after_each_epoch=False):

        flags = tf.app.flags.FLAGS
        train_x, train_y = self.get_train_ds()

        # profiler_hook = tf.train.ProfilerHook(
        #     save_steps=50,
        #     output_dir=flags.model_dir + '/train'
        # )

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_x},
            y=train_y,
            batch_size=flags.train_batch_size,
            num_epochs=None,
            shuffle=True
        )

        for i in range(epochs):
            start_time = time.time()
            self.estimator.train(
                input_fn=train_input_fn,
                steps=steps,
                # hooks=[profiler_hook]
            )
            duration = round(time.time() - start_time, 3)
            self.total_train_duration += duration

            print("Training duration: " + common.duration_to_string(duration))
            print("Training epoch %d of %d completed" % (i+1, epochs))

            if eval_after_each_epoch:
                self.eval()

    def eval(self):

        flags = tf.app.flags.FLAGS
        eval_x, eval_y = self.get_eval_ds()
        # logging_hook = tf.train.LoggingTensorHook(
        #     tensors={
        #         # "probabilities": "softmax_tensor",
        #         "pred": "diff"
        #     },
        #     every_n_iter=1
        # )

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_x},
            y=eval_y,
            batch_size=flags.eval_batch_size,
            shuffle=False
        )

        start_time = time.time()
        result = self.estimator.evaluate(
            input_fn=eval_input_fn,
            # hooks=[logging_hook]
        )
        self.eval_results.append(result)

        duration = round(time.time() - start_time, 3)
        self.total_eval_duration += duration

        print("Eval duration: " + common.duration_to_string(duration))
        print("Eval result:", result)



def model_fn(features, labels, mode, params, config):
    """Model function for CNN."""
    model_details["model_dir"] = config.model_dir

    # Input Layer
    with tf.name_scope("input_layer"):
        input_layer = features["x"]

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[7, 7],
        strides=2,
        padding="same",
        activation=tf.nn.relu,
        name="conv3x3"
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=2,
        padding="same",
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
        filters=172,
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

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu,
        name="conv4"
    )
    # pool4 = tf.layers.max_pooling2d(
    #     inputs=conv4,
    #     pool_size=[2, 2],
    #     strides=2,
    #     name="pool4"
    # )

    pool_flat = tf.reshape(conv4, [-1, conv4.shape[1]*conv4.shape[2]*conv4.shape[3]], name="pool_flat")

    # Dense Layer
    dense1 = tf.layers.dense(
            inputs=pool_flat, units=1024, activation=tf.nn.relu, name="dense1")

    dense2 = tf.layers.dense(
            inputs=dense1, units=512, activation=tf.nn.relu, name="dense2")

    dense3 = tf.layers.dense(
        inputs=dense1, units=256, activation=tf.nn.relu, name="dense3")

    # GET MODEL DETAILS
    model_details["params"]["conv1"] = {"shape": conv1.shape.as_list()}
    model_details["params"]["conv2"] = {"shape": conv2.shape.as_list()}
    model_details["params"]["conv3"] = {"shape": conv3.shape.as_list()}
    model_details["params"]["conv4"] = {"shape": conv4.shape.as_list()}
    model_details["params"]["pool1"] = {"shape": pool1.shape.as_list()}
    model_details["params"]["pool2"] = {"shape": pool2.shape.as_list()}
    model_details["params"]["pool3"] = {"shape": pool3.shape.as_list()}
    # model_details["params"]["pool4"] = {"shape": pool4.shape.as_list()}
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
        learning_rate = common.get_learning_rate_from_flags(tf.app.flags.FLAGS)

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


def load_stl10_train_dataset():
    return ds.load_dataset_from_pickles([
        "/datasets/stl10/original_train.pkl",
        "/datasets/stl10/mirror_train.pkl",
        "/datasets/stl10/rand_distorted_train_0.pkl",
        "/datasets/stl10/rand_distorted_train_1.pkl",
        "/datasets/stl10/rand_distorted_train_2.pkl",
    ])


def load_stl10_eval_dataset():
    return ds.load_dataset_from_pickles([
        "/datasets/stl10/original_test.pkl",
    ])


def build_stl10_app_flags():
    # Hyperparameters for the model training/evaluation
    # They are accessible everywhere in the application via tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string("model_dir", "../models/stl10/test1")
    tf.app.flags.DEFINE_float("dropout_rate", 0.4)
    tf.app.flags.DEFINE_integer("eval_batch_size", 256)
    tf.app.flags.DEFINE_integer("training_batch_size", 128)
    tf.app.flags.DEFINE_float("initial_learning_rate", 0.05)
    tf.app.flags.DEFINE_bool("use_static_learning_rate", False)
    tf.app.flags.DEFINE_float("learning_rate_decay_rate", 0.96)
    tf.app.flags.DEFINE_integer("learning_rate_decay_steps", 5000)


def main(unused_argv):

    build_stl10_app_flags()

    classifier = Classifier(
        model_fn=model_fn,
        model_params={},
        load_train_ds_fn=load_stl10_train_dataset,
        load_eval_ds_fn=load_stl10_eval_dataset,
        clean_old_model_data=False,
    )

    classifier.train(
        steps=2000,
        epochs=1,
        eval_after_each_epoch=True,
    )



    model_stats_map = {}
    for params_name, params in model_configs.items():


        eval_results = []
        total_train_duration = 0
        total_eval_duration = 0
        for i in range(TRAINING_EPOCHS):
            train_duration = train_model(classifier)
            total_train_duration += train_duration

            eval_result, eval_duration = eval_model(classifier)
            eval_results.append(eval_result)
            total_eval_duration += eval_duration

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
