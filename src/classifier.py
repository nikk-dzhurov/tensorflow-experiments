from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pprint
import numpy as np
from PIL import Image
import tensorflow as tf

import hooks
import common
import files
import image
from image import LabeledImage


class Classifier(object):
    def __init__(self, model_fn, model_params, class_names):

        self.validate_required_app_flags()

        self.class_names = class_names

        self._eval_ds = None
        self._train_ds = None
        self._predict_ds = None
        self.model_details = {
            "model_flags": tf.app.flags.FLAGS.flag_values_dict(),
            "model_params": model_params,
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

        self._eval_columns = [
            ('global_step', 'Global Step'),
            ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
        ]

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
            "image_width": {"type": int, "range": [28, 128]},
            "image_height": {"type": int, "range": [28, 128]},
            "image_channels": {"type": int, "range": [1, 4]},
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

    @staticmethod
    def print_results_as_table(columns, data):
        total_len = 0
        col_lens = {}
        columns_map = {}

        for field in columns:
            field_key = field[0]
            field_name = field[1]
            columns_map[field_key] = field_name
            v_len = max(len(field_name) + 1, 10)
            col_lens[field_key] = v_len
            total_len += v_len

        def row_string(data):
            res = ""
            for field in columns:
                field_key = field[0]
                res += "|"
                res += col_string(data[field_key], " ", col_lens[field_key])
            return res + "|"

        def col_string(data, pad_symbol, pad_num):
            if type(data) is float or type(data) is np.float32:
                return "{:{}<{}.6f}".format(data, pad_symbol, pad_num)
            return "{:{}<{}}".format(data, pad_symbol, pad_num)

        def separator_string():
            res = ""
            for field in columns:
                field_key = field[0]
                res += "+"
                res += col_string("", "-", col_lens[field_key])
            return res + "+"

        top_bottom_line = "+{:-<{}}+".format('', total_len + len(columns) - 1)

        print(top_bottom_line)
        print(row_string(columns_map))
        for res in data:
            print(separator_string())
            print(row_string(res))
        print(top_bottom_line)

    def train(self,
              steps,
              load_train_ds_fn,
              epochs=1,
              clean_old_model_data=False,
              load_eval_ds_fn=None,
              eval_after_each_epoch=False):

        flags = tf.app.flags.FLAGS

        if type(steps) is not int or 1 < steps > 10000:
            raise Exception("Invalid steps argument")
        if type(epochs) is not int or 1 < steps > 100:
            raise Exception("Invalid epochs argument")
        if not callable(load_train_ds_fn):
            raise Exception("load_train_ds_fn argument is not callable")
        if eval_after_each_epoch and not callable(load_eval_ds_fn):
            raise Exception("load_eval_ds_fn argument is not callable")

        if clean_old_model_data:
            files.clean_dir(flags.model_dir)

        train_x, train_y = self._get_train_ds(load_train_ds_fn)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_x},
            y=train_y,
            batch_size=flags.train_batch_size,
            num_epochs=None,
            shuffle=True
        )

        print("Train {}x{} steps".format(epochs, steps))

        for i in range(epochs):
            start_time = time.time()
            self._estimator.train(
                input_fn=train_input_fn,
                steps=steps,
                hooks=[]
            )
            duration = round(time.time() - start_time, 3)
            self.total_train_duration += duration

            print("Training duration: " + common.duration_to_string(duration))
            print("Training epoch {} of {} completed".format(i+1, epochs))

            if eval_after_each_epoch:
                self.eval(load_eval_ds_fn, save_eval_map=False)

        self._save_results()

    def eval(self, load_eval_ds_fn, save_eval_map=True, log_tensors=False):
        flags = tf.app.flags.FLAGS

        if not callable(load_eval_ds_fn):
            raise Exception("load_eval_ds_fn argument is not callable")

        eval_x, eval_y = self._get_eval_ds(load_eval_ds_fn)

        eval_hooks = []
        if save_eval_map:
            eval_hooks.append(hooks.EvaluationMapSaverHook(
                tensor_names=["labels", "predictions", "top_k_values", "top_k_indices"],
            ))
        if log_tensors:
            eval_hooks.append(tf.train.LoggingTensorHook(
                tensors={
                    "probabilities": "softmax_tensor",
                },
                every_n_iter=1,
            ))

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_x},
            y=eval_y,
            batch_size=flags.eval_batch_size,
            shuffle=False,
        )

        start_time = time.time()
        result = self._estimator.evaluate(
            input_fn=eval_input_fn,
            hooks=eval_hooks
        )
        self.eval_results.append(result)

        duration = round(time.time() - start_time, 3)
        self.total_eval_duration += duration

        print("Eval duration: " + common.duration_to_string(duration))
        self.print_results_as_table(self._eval_columns, [result])

    def predict(self, image_location="/test_images/plane2.jpg"):
        app_flags = tf.app.flags.FLAGS
        if type(image_location) is not str or image_location == "":
            raise ValueError("Specify valid image location")

        img = Image.open(image_location).convert('RGB')
        if img.width != app_flags.image_width or img.height != app_flags.image_height:
            raise ValueError("Please provide image with dimensions: {}x{}"
                             .format(app_flags.image_width, app_flags.image_height))

        images = common.prepare_images([np.array(img)])
        actual_class_name = "airplane"

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": images},
            y=None,
            batch_size=1,
            shuffle=False
        )

        pred_generator = self._estimator.predict(input_fn=input_fn)

        for res in pred_generator:
            print(res)
            prediction_class = self._index_to_class_name(res["class"])
            probability = res["probabilities"][res["class"]]*100

            print("Prediction: %s(%.2f%%), actual class %s" % (prediction_class, probability,  actual_class_name))
            # print(tf.Session().run(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=res["logits"])))

        # LabeledImage(img, "bird", max_value=255).save()
        # print(img.shape)
        # pass

    def get_final_eval_result(self):
        if self.eval_results is not None and len(self.eval_results) > 0:
            res = self.eval_results[-1].copy()
            res["accuracy"] = res["accuracy"].item()
            res["loss"] = res["loss"].item()
            res["global_step"] = res["global_step"].item()
        else:
            res = {"error": "eval_results are missing"}

        return res

    def _get_eval_ds(self, load_fn):
        if self._eval_ds is None:
            self._eval_ds = load_fn()
            # self._eval_ds = (self._eval_ds[0][0:10], self._eval_ds[1][0:10])  # Useful for development
            print("Evaluation dataset loaded successfully!")
            self.print_ds_details(self._eval_ds, "eval")

        return self._eval_ds

    def _get_train_ds(self, get_fn):
        if self._train_ds is None:
            self._train_ds = get_fn()
            print("Training dataset loaded successfully!")
            self.print_ds_details(self._train_ds, "train")

        return self._train_ds

    def _get_predict_ds(self, load_fn):
        self._predict_ds = load_fn()
        print("Prediction dataset loaded successfully!")
        self.print_ds_details(self._predict_ds, "predict")

        return self._predict_ds

    def _class_name_to_index(self, class_name):
        return self.class_names.index(class_name)

    def _index_to_class_name(self, idx):
        return self.class_names[idx]

    def _save_results(self):
        pp = pprint.PrettyPrinter(indent=2, compact=True)

        for name in self._estimator.get_variable_names():
            if name.find("optimizer") != -1 or name in ["beta1_power", "beta2_power"]:
                continue

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
            "final_result": self.get_final_eval_result(),
            "total_train_duration": common.duration_to_string(self.total_train_duration),
            "total_eval_duration": common.duration_to_string(self.total_eval_duration),
        }

        files.save_pickle(
            model_stats_map,
            os.path.join(tf.app.flags.FLAGS.model_dir, "last_result.pkl")
        )
        files.save_json(
            model_stats_map,
            os.path.join(tf.app.flags.FLAGS.model_dir, "last_result.json")
        )

        self.print_results_as_table(self._eval_columns, self.eval_results)
        pp.pprint(model_stats_map)
        print("Total training duration: " + common.duration_to_string(self.total_train_duration))
        print("Total evaluation duration: " + common.duration_to_string(self.total_eval_duration))

