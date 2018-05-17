from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import tensorflow as tf

import common


class Classifier(object):
    def __init__(self, model_fn, model_params):

        self.validate_required_app_flags()

        self._eval_ds = None
        self._train_ds = None
        self._predict_ds = None
        self.model_details = {
            "model_dir": tf.app.flags.FLAGS.model_dir,
            "model_vars": {}
        }
        self.eval_results = []
        self.total_eval_duration = 0
        self.total_train_duration = 0

        self._estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=tf.app.flags.FLAGS.model_dir,
            params=model_params,
            config=self.get_run_config_from_flags(),
        )

    @staticmethod
    def validate_required_app_flags():
        app_flags = tf.app.flags.FLAGS
        required_flags = {
            "model_dir": {"type": str},
            "dropout_rate": {"type": float, "range": [0.1, 1.0]},
            "eval_batch_size": {"type": int, "range": [1, 1024]},
            "train_batch_size": {"type": int, "range": [1, 1024]},
            "initial_learning_rate": {"type": float, "range": [0.0001, 0.15]},
            "use_static_learning_rate": {"type": bool},
            "learning_rate_decay_rate": {"type": float, "range": [0.8, 0.99]},
            "learning_rate_decay_steps": {"type": int, "range": [10, 10000]},
            "ignore_gpu": {"type": bool},
            "per_process_gpu_memory_fraction": {"type": float, "range": [0.1, 1.0]},
        }

        for flag_name, validations in required_flags.items():
            flag_value = app_flags.get_flag_value(flag_name, None)
            if flag_value is None:
                raise Exception("Flag \"{}\" is not defined or has None value".format(flag_name))

            if type(flag_value) is not validations["type"]:
                raise Exception("Flag \"{}\" should be of type: {}, received: {}".format(flag_name, validations["type"], type(flag_value)))

            if validations["type"] is str and flag_value == "":
                raise Exception("Flag \"{}\" should not be empty string".format(flag_name))

            if validations["type"] is int or validations["type"] is float:
                if flag_value < validations["range"][0] or flag_value > validations["range"][1]:
                    raise Exception("Flag \"{}\" should be in range({}, {}), received: {}".format(flag_name, validations["range"][0], validations["range"][1], flag_value))

    @staticmethod
    def print_ds_details(ds, ds_name="dataset"):
        print("{}_x:\n\t shape: {}\n\t type: {}".format(ds_name, ds[0].shape, ds[0].dtype))
        print("{}_y:\n\t shape: {}\n\t type: {}".format(ds_name, ds[1].shape, ds[1].dtype))

    @staticmethod
    def get_run_config_from_flags():
        flags = tf.app.flags.FLAGS
        sess_config = tf.ConfigProto()

        if flags.ignore_gpu:
            sess_config = tf.ConfigProto(device_count={'GPU': 0})
        elif flags.per_process_gpu_memory_fraction != 1.0:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=flags.per_process_gpu_memory_fraction)
            sess_config = tf.ConfigProto(gpu_options=gpu_options)

        return tf.estimator.RunConfig(session_config=sess_config)

    def _get_eval_ds(self, load_fn):
        if self._eval_ds is None:
            self._eval_ds = load_fn()
            print("Eval dataset loaded successfully!")
            self.print_ds_details(self._eval_ds, "eval")

        return self._eval_ds

    def _get_train_ds(self, get_fn):
        if self._train_ds is None:
            self._train_ds = get_fn()
            print("Train dataset loaded successfully!")
            self.print_ds_details(self._train_ds, "train")

        return self._train_ds

    def _get_predict_ds(self, load_fn):
        self._predict_ds = load_fn()
        print("Predict dataset loaded successfully!")
        self.print_ds_details(self._predict_ds, "predict")

        return self._predict_ds

    def _save_results(self):
        pp = pprint.PrettyPrinter(indent=2, compact=True)

        for name in self._estimator.get_variable_names():
            val = self._estimator.get_variable_value(name)
            if len(val.shape) > 0:
                self.model_details["model_vars"][name] = {
                    "shape": val.shape,
                }
            else:
                self.model_details["model_vars"][name] = {
                    "value": val.item(),
                }

        model_stats_map = {
            "model_details": self.model_details,
            "final_result": common.get_final_eval_result(self.eval_results),
            "total_train_duration": common.duration_to_string(self.total_train_duration),
            "total_eval_duration": common.duration_to_string(self.total_eval_duration),
        }

        common.save_pickle(
            model_stats_map,
            os.path.join(tf.app.flags.FLAGS.model_dir, "last_result.pkl")
        )
        common.save_json(
            model_stats_map,
            os.path.join(tf.app.flags.FLAGS.model_dir, "last_result.json")
        )

        pp.pprint(model_stats_map)
        print("Total training duration: " + common.duration_to_string(self.total_train_duration))
        print("Total evaluation duration: " + common.duration_to_string(self.total_eval_duration))

    def train(self, steps, load_train_ds_fn, clean_old_model_data=False, load_eval_ds_fn=None, epochs=1, eval_after_each_epoch=False):
        flags = tf.app.flags.FLAGS

        if type(steps) is not int or steps < 1 or steps > 10000:
            raise Exception("Invalid steps argument")
        if type(epochs) is not int or epochs < 1 or epochs > 100:
            raise Exception("Invalid epochs argument")
        if not callable(load_train_ds_fn):
            raise Exception("load_train_ds_fn argument is not callable")
        if eval_after_each_epoch and not callable(load_eval_ds_fn):
            raise Exception("load_eval_ds_fn argument is not callable")

        if clean_old_model_data:
            common.clean_dir(flags.model_dir)

        train_x, train_y = self._get_train_ds(load_train_ds_fn)

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
            self._estimator.train(
                input_fn=train_input_fn,
                steps=steps,
                # hooks=[profiler_hook]
            )
            duration = round(time.time() - start_time, 3)
            self.total_train_duration += duration

            print("Training duration: " + common.duration_to_string(duration))
            print("Training epoch {} of {} completed".format(i+1, epochs))

            if eval_after_each_epoch:
                self.eval(load_eval_ds_fn)

        self._save_results()

    def eval(self, load_eval_ds_fn):
        flags = tf.app.flags.FLAGS

        if not callable(load_eval_ds_fn):
            raise Exception("load_eval_ds_fn argument is not callable")

        eval_x, eval_y = self._get_eval_ds(load_eval_ds_fn)
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
        result = self._estimator.evaluate(
            input_fn=eval_input_fn,
            # hooks=[logging_hook]
        )
        self.eval_results.append(result)

        duration = round(time.time() - start_time, 3)
        self.total_eval_duration += duration

        print("Eval duration: " + common.duration_to_string(duration))
        print("Eval result:", result)

    def predict(self, image_location=""):
        print("Not implemented!")
        pass