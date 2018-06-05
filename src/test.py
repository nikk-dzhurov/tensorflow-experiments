from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import tensorflow as tf

import common
import image_dataset as ds

print("Just test")

import numpy
import tensorflow as tf
from random import randint
from google.protobuf import text_format

from tensorflow.python.saved_model import tag_constants

if __name__ == "__main__":

    def _parse_input_graph_proto(input_graph, input_binary):
        """Parser input tensorflow graph into GraphDef proto."""
        if not tf.gfile.Exists(input_graph):
            print("Input graph file '" + input_graph + "' does not exist!")
            return -1
        input_graph_def = tf.GraphDef()
        mode = "rb" if input_binary else "r"
        with tf.gfile.FastGFile(input_graph, mode) as f:
            if input_binary:
                input_graph_def.ParseFromString(f.read())
            else:
                text_format.Merge(f.read(), input_graph_def)
        return input_graph_def


    def _parse_input_meta_graph_proto(input_graph, input_binary):
        """Parser input tensorflow graph into MetaGraphDef proto."""
        if not tf.gfile.Exists(input_graph):
            print("Input meta graph file '" + input_graph + "' does not exist!")
            return -1
        input_meta_graph_def = tf.MetaGraphDef()
        mode = "rb" if input_binary else "r"
        with tf.gfile.FastGFile(input_graph, mode) as f:
            if input_binary:
                input_meta_graph_def.ParseFromString(f.read())
            else:
                text_format.Merge(f.read(), input_meta_graph_def)
        print("Loaded meta graph file '" + input_graph)
        return input_meta_graph_def

    graph = _parse_input_graph_proto("../models/stl10/adamOp_1_eps0.1_lr_0.0005/graph.pbtxt", False)
    # # graph2 = _parse_input_meta_graph_proto("../models/stl10/adamOp_1_eps0.1_lr_0.0005/model.ckpt-4502", False)
    #
    # print(graph)


    def fix_graph_def(graph_def):
        # fix nodes
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            if "dilations" in node.attr:
                del node.attr["dilations"]
            if "index_type" in node.attr:
                del node.attr["index_type"]


    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("../models/stl10/adamOp_1_eps0.1_lr_0.0005/model.ckpt-5000.meta")
        saver.restore(sess, '../models/stl10/adamOp_1_eps0.1_lr_0.0005/model.ckpt-5000')

        saver0 = tf.train.Saver()
        saver0.save(sess, 'models/my-model-10000')
        # Generates MetaGraphDef.
        saver0.export_meta_graph('models/my-model-10000.meta', strip_default_attrs=True)
    #
    # tf.train.export_meta_graph("../models/asd-meta", gra)
    #
    fix_graph_def(graph)
    # # fix_graph_def(graph2)
    # #
    tf.train.write_graph(graph, logdir="./models/", name="graph.pbtxt", as_text=True)
    # # tf.train.write_graph(graph2, logdir="../models/stl10/adamOp_1_eps0.1_lr_0.0005/", name="model.ckpt-4502.meta", as_text=True)