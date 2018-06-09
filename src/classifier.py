import os
import time
import pprint
import numpy as np
from PIL import Image
import tensorflow as tf

import file
import hooks
import image_dataset


class Classifier(object):
    """
    Classifier class for train, prediction and evaluation

    Classifier object create tf.estimator.Estimator from received dataset module.
    It provides functions for train, prediction and evaluation.
    It provides functions for exporting model
    """

    def __init__(self, ds_module):
        """Initialize/Construct Classifier object"""

        self.validate_ds_module(ds_module)

        ds_module.build_app_flags()
        self.validate_required_app_flags()

        self._eval_ds = None
        self._train_ds = None
        self._predict_ds = None
        self.model_details = {
            "model_flags": tf.app.flags.FLAGS.flag_values_dict(),
            "model_params": ds_module.get_model_params(),
            "model_vars": {}
        }
        self.eval_results = []
        self.total_eval_duration = 0
        self.total_train_duration = 0

        self.ds_module = ds_module
        self.class_names = ds_module.get_class_names()

        self._estimator = tf.estimator.Estimator(
            model_fn=ds_module.model_fn,
            model_dir=tf.app.flags.FLAGS.model_dir,
            params=ds_module.get_model_params(),
            config=self.get_run_config_from_flags(),
        )

        self._eval_columns = [
            ('global_step', 'Global Step'),
            ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
        ]

    @staticmethod
    def validate_ds_module(ds_module):
        required_functions = [
            "build_app_flags",
            "model_fn",
            "load_eval_dataset",
            "load_train_dataset",
            "get_model_params",
            "get_class_names"
        ]

        for func_name in required_functions:
            func = getattr(ds_module, func_name, None)
            if not callable(func):
                raise Exception(
                    "Dataset module does not provide function with name {}".format(func_name))

    @staticmethod
    def validate_required_app_flags():
        """Validate required tf.app.flags.FLAGS and raise ValueError if validation fails"""

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
                raise Exception("Flag \"{}\" should be of type: {}, received: {}"
                                .format(flag_name, validations["type"], type(flag_value)))

            if validations["type"] is str and flag_value == "":
                raise Exception("Flag \"{}\" should not be empty string".format(flag_name))

            if validations["type"] is int or validations["type"] is float:
                if flag_value < validations["range"][0] or flag_value > validations["range"][1]:
                    raise Exception("Flag \"{}\" should be in range({}, {}), received: {}"
                                    .format(flag_name, validations["range"][0],
                                            validations["range"][1], flag_value))

    @staticmethod
    def get_run_config_from_flags():
        """Build tf.estimator.RunConfig object from application flags"""

        flags = tf.app.flags.FLAGS
        sess_config = tf.ConfigProto()

        if flags.ignore_gpu:
            sess_config = tf.ConfigProto(device_count={'GPU': 0})
        elif flags.per_process_gpu_memory_fraction != 1.0:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=flags.per_process_gpu_memory_fraction)
            sess_config = tf.ConfigProto(gpu_options=gpu_options)

        return tf.estimator.RunConfig(
            session_config=sess_config,
            log_step_count_steps=100,
            save_summary_steps=100,
            save_checkpoints_steps=1000,
            keep_checkpoint_max=30,
        )

    def export_model(self):
        """Export trained model as SavedModel that could be used in production"""

        self._estimator.export_savedmodel(
            export_dir_base=tf.app.flags.FLAGS.model_dir,
            serving_input_receiver_fn=tf.estimator.export.build_parsing_serving_input_receiver_fn({
                "x": tf.FixedLenFeature(shape=(96, 96, 3), dtype=tf.float32),
            }),
            strip_default_attrs=True,
        )

    def train(self,
              steps,
              epochs=1,
              clean_old_model_data=False,
              eval_after_each_epoch=False):
        """Train model function, that allow evaluation after each training epoch"""

        flags = tf.app.flags.FLAGS

        if type(steps) is not int or 1 < steps > 10000:
            raise Exception("Invalid steps argument")
        if type(epochs) is not int or 1 < epochs > 100:
            raise Exception("Invalid epochs argument")

        if clean_old_model_data:
            file.clean_dir(flags.model_dir)

        train_x, train_y = self._get_train_ds()

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

            print("Training duration: " + self._duration_to_string(duration))
            print("Training epoch {} of {} completed".format(i+1, epochs))

            if eval_after_each_epoch:
                self.eval(save_eval_map=False)

        self._save_results()

    def eval(self, save_eval_map=True, log_tensors=True):
        """Evaluate model function"""

        flags = tf.app.flags.FLAGS

        eval_x, eval_y = self._get_eval_ds()

        eval_hooks = []
        if save_eval_map:
            eval_hooks.append(hooks.EvaluationMapSaverHook(
                tensor_names=[
                    "labels",
                    "predictions/ArgMax",
                    "top_k/indices",
                    "top_k/values"
                ],
            ))
        if log_tensors:
            eval_hooks.append(tf.train.LoggingTensorHook(
                tensors={
                    "loss": "loss/value",
                },
                every_n_iter=5,
            ))

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_x},
            y=eval_y,
            num_epochs=1,
            batch_size=flags.eval_batch_size,
            shuffle=False,
        )

        eval_steps = len(eval_y) // flags.eval_batch_size
        if len(eval_y) % flags.eval_batch_size > 0:
            eval_steps += 1

        start_time = time.time()

        result = self._estimator.evaluate(
            input_fn=eval_input_fn,
            steps=eval_steps,
            hooks=eval_hooks,
        )

        duration = round(time.time() - start_time, 3)
        self.total_eval_duration += duration

        self.eval_results.append(result)
        results_table = self.format_results_as_table_string(self._eval_columns, [result])

        print(results_table)
        print("Eval duration: " + self._duration_to_string(duration))

    def predict(self):
        """Make predictions for given dataset"""

        flags = tf.app.flags.FLAGS

        pred_x, pred_y = self._get_eval_ds()

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": pred_x},
            y=pred_y,
            batch_size=flags.eval_batch_size,
            shuffle=False,
        )

        pred_generator = self._estimator.predict(input_fn=input_fn)

        for idx, res in enumerate(pred_generator):
            prediction_class = self._index_to_class_name(res["class"])
            probability = res["probabilities"][res["class"]]*100
            actual_class_name = self._index_to_class_name(pred_y[idx])

            print("Prediction for index: %d - %s(%.2f%%), actual class %s" %
                  (idx, prediction_class, probability,  actual_class_name))

    def predict_image_label(self, image_location, expected_label):
        """Make prediction for single image from given location"""

        app_flags = tf.app.flags.FLAGS
        if type(image_location) is not str or image_location == "":
            raise ValueError("Specify valid image location")
        if type(expected_label) is not str or expected_label == "":
            raise ValueError("Specify valid value for actual_label")

        img = Image.open(image_location).convert('RGB')
        if img.width != app_flags.image_width or img.height != app_flags.image_height:
            raise ValueError("Please provide image with dimensions: {}x{}"
                             .format(app_flags.image_width, app_flags.image_height))

        images = image_dataset.prepare_images([np.array(img)])

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

            print("Prediction: %s(%.2f%%), expected label %s" % (prediction_class, probability,  expected_label))
            # print(tf.Session().run(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=res["logits"])))

        # LabeledImage(img, "bird", max_value=255).save()
        # print(img.shape)
        # pass

    def get_final_eval_result(self):
        """Get last result from evaluation results and make it JSON serializable"""

        if self.eval_results is not None and len(self.eval_results) > 0:
            res = self.eval_results[-1].copy()
            res["accuracy"] = res["accuracy"].item()
            res["loss"] = res["loss"].item()
            res["global_step"] = res["global_step"].item()
        else:
            res = {"error": "eval_results are missing"}

        return res

    @staticmethod
    def format_results_as_table_string(columns, data):
        """Format array of dictionaries as table by given columns"""

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

            for f in columns:
                f_key = f[0]
                res += "|"
                res += col_string(data[f_key], " ", col_lens[f_key])

            return res + "|\n"

        def col_string(data, pad_symbol, pad_num):
            if type(data) is float or type(data) is np.float32:
                return "{:{}<{}.6f}".format(data, pad_symbol, pad_num)

            return "{:{}<{}}".format(data, pad_symbol, pad_num)

        def separator_string():
            res = ""

            for f in columns:
                f_key = f[0]
                res += "+"
                res += col_string("", "-", col_lens[f_key])

            return res + "+\n"

        top_bottom_line = "+{:-<{}}+\n".format('', total_len + len(columns) - 1)

        result = top_bottom_line
        result += row_string(columns_map)

        for res in data:
            result += separator_string()
            result += row_string(res)

        result += top_bottom_line

        return result

    @staticmethod
    def _print_ds_details(ds, ds_name="dataset"):
        """Pretty print dataset details"""

        print("{}_x:\n\t shape: {}\n\t type: {}".format(ds_name, ds[0].shape, ds[0].dtype))
        print("{}_y:\n\t shape: {}\n\t type: {}".format(ds_name, ds[1].shape, ds[1].dtype))

    @staticmethod
    def _print_ds_loaded(name):
        print("{} dataset loaded successfully!".format(name))

    @staticmethod
    def _duration_to_string(dur_in_sec=0):
        """Convert duration(int) to string"""

        days, remainder = divmod(dur_in_sec, 60 * 60 * 24)
        hours, remainder = divmod(remainder, 60 * 60)
        minutes, seconds = divmod(remainder, 60)
        output = ""
        if days > 0:
            output += "%d days, " % days
        if hours > 0:
            output = "%d hours, " % hours
        if minutes > 0:
            output += "%d min, " % minutes
        if seconds > 0 or len(output) == 0:
            output += "%.3f sec" % seconds
        if output[-2:] == ", ":
            output = output[:-2]

        return output

    def _get_eval_ds(self):
        """Lazy load evaluation dataset"""

        if self._eval_ds is None:
            self._eval_ds = self.ds_module.load_eval_dataset()
            # self._eval_ds = (self._eval_ds[0][0:10], self._eval_ds[1][0:10])  # Useful for development
            self._print_ds_loaded("Evaluation")
            self._print_ds_details(self._eval_ds, "eval")

        return self._eval_ds

    def _get_train_ds(self):
        """Lazy load training dataset"""

        if self._train_ds is None:
            self._train_ds = self.ds_module.load_train_dataset()
            self._print_ds_loaded("Training")
            self._print_ds_details(self._train_ds, "train")

        return self._train_ds

    def _class_name_to_index(self, class_name):
        """Convert class name to label index"""

        return self.class_names.index(class_name)

    def _index_to_class_name(self, idx):
        """Convert label index to class name"""

        return self.class_names[idx]

    def _save_results(self):
        """Export latest results from model training/evaluation in pickle and JSON formats"""

        pp = pprint.PrettyPrinter(indent=2, compact=True)

        for name in self._estimator.get_variable_names():
            if name == "h" or name.find("optimizer") != -1 or name in ["beta1_power", "help", "beta2_power"]:
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
            "total_train_duration": self._duration_to_string(self.total_train_duration),
            "total_eval_duration": self._duration_to_string(self.total_eval_duration),
        }

        results_table = self.format_results_as_table_string(self._eval_columns, self.eval_results)

        file.save_pickle(
            model_stats_map,
            os.path.join(tf.app.flags.FLAGS.model_dir, "last_result.pkl")
        )
        file.save_json(
            model_stats_map,
            os.path.join(tf.app.flags.FLAGS.model_dir, "last_result.json")
        )
        file.save_txt(
            results_table,
            os.path.join(tf.app.flags.FLAGS.model_dir, "last_results_table.txt")
        )

        print(results_table)
        pp.pprint(model_stats_map)
        print("Total training duration: " + self._duration_to_string(self.total_train_duration))
        print("Total evaluation duration: " + self._duration_to_string(self.total_eval_duration))
