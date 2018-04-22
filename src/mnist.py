from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import clean_dir
from common import build_layer_summaries

import numpy as np
import tensorflow as tf

MODEL_DIR = "../models/mnist"
VALIDATION_DATA_SIZE = 10000


tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SIZE = 28

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    with tf.name_scope("input_layer"):
        input_layer = tf.reshape(features["x"], [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv1"
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2,
        name="pool1"
    )

    norm1 = tf.nn.local_response_normalization(pool1, 4, alpha=0.00011, beta=0.75, name="norm1")

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=norm1,
        filters=64,
        kernel_size=[5, 5],
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

    norm2 = tf.nn.local_response_normalization(pool2, 4, alpha=0.00011, beta=0.75, name="norm2")

    build_layer_summaries("conv1")
    build_layer_summaries("conv2")

    # Dense Layer
    with tf.name_scope("dense"):
        pool2_flat = tf.reshape(norm2, [-1, 7 * 7 * 64], name="pool2_reshape")
        dense = tf.layers.dense(
            inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense")
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout')

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10, name="logits")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name="predictions"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope="calc_loss")
    tf.summary.scalar("cross_entropy", loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        summary_saver = tf.train.SummarySaverHook(
            save_steps=1, output_dir=MODEL_DIR + "/train", summary_op=tf.summary.merge_all())
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.001, name="gradient_descent_optimizer")
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            name="minimize_loss")
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_saver])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # clean_dir(MODEL_DIR)

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=MODEL_DIR)

    # # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=10)
    # Train the model

    # profiler_hook = tf.train.ProfilerHook(save_steps=50, output_dir=MODEL_DIR + '/train')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=500,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=5000,
        # hooks=[logging_hook, profiler_hook]
    )

    # Evaluate the model and print results
    # print("LEN?>", len(eval_data),  len(eval_labels))

    batch_size = 1000
    eval_steps = VALIDATION_DATA_SIZE // batch_size
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=batch_size,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    print("Eval result:", eval_results)


if __name__ == "__main__":
    tf.app.run(main=main)
