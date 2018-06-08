import os
import numpy as np
import tensorflow as tf

import file
from image import LabeledImage


class EvaluationMapSaverHook(tf.train.SessionRunHook):
    """
    EvaluationMapSaverHook class for extracting evaluation tensors
    It extends tf.train.SessionRunHook to access tensors in the current session
    """

    def __init__(self, tensor_names=None, file_name="eval_map.pkl"):
        """Initialize/Construct EvaluationMapSaverHook object"""

        if tensor_names is None or len(tensor_names) == 0:
            raise ValueError("tensor_names should has at least 1 element")
        if type(file_name) is not str or file_name == "":
            raise ValueError("file_location should be valid string")

        self._iter_count = None
        self._tensors = None
        self._should_trigger = False
        self._results = {}
        self._file_name = file_name
        self._tensor_names = tensor_names
        self._timer = tf.train.SecondOrStepTimer(every_steps=1)

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        self._tensors = [_get_graph_element(t_name) for t_name in self._tensor_names]

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return tf.train.SessionRunArgs(self._tensors)
        else:
            return None

    def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
        if self._should_trigger:
            for idx, t_name in enumerate(self._tensor_names):
                if self._results.get(t_name, None) is None:
                    self._results[t_name] = run_values.results[idx]
                else:
                    self._results[t_name] = np.concatenate([
                        self._results[t_name], run_values.results[idx]], axis=0)

        self._iter_count += 1

    def end(self, session):
        file.save_pickle(
            self._results,
            os.path.join(tf.app.flags.FLAGS.model_dir, self._file_name)
        )


def _get_graph_element(tensor_name):
    """Extract tensor from the model's graph"""

    if type(tensor_name) is not str:
        raise ValueError("Passed argument %s should be of type string".format(tensor_name))

    graph = tf.get_default_graph()
    if ":" in tensor_name:
        element = graph.as_graph_element(tensor_name)
    else:
        element = graph.as_graph_element(tensor_name + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(tensor_name + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, as this `Operation` has multiple outputs"
                             .format(tensor_name))

    return element
