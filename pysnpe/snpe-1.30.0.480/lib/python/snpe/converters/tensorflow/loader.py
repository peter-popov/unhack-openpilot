#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os

import tensorflow as tf
from tensorflow.python.framework import graph_util

import snpe.converters.common.utils.code_to_message as code_to_message
from snpe.converters.tensorflow.util import ConverterError
from snpe.converters.tensorflow.util import GraphHelper
from snpe.converters.tensorflow.util import GraphPrinter
from snpe.converters.tensorflow.util import VisitableGraph


class ModelLoader(object):
    def __init__(self, logger):
        """
        :type logger: logging.Logger
        """
        self._logger = logger

    def load(self, graph_pb_or_meta_path, input_nodes_names, input_nodes_shapes, input_nodes_types, out_node_names,
             session):
        """
        Loads the Tensorflow Graph into the specified Session's Graph and builds a Model instance
        with all the relevant information for a ModelConverter to use during conversion.
        :type graph_pb_or_meta_path: str
        :type input_nodes_names: list[str]
        :type input_nodes_shapes: list[str]
        :type input_nodes_types: list[str]
        :type out_node_names: list[str]
        :type session: tensorflow.Session
        :rtype: Model
        """
        if len(input_nodes_names) != len(input_nodes_shapes):
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_INPUT_NODE_SHAPE_DIMS_MISMATCH'))
        if input_nodes_types is not None and len(input_nodes_types):
            if len(input_nodes_names) != len(input_nodes_types):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_INPUT_TYPES_AND_NAMES_NOT_IN_PAIRS'))
        else:
            # Set all types to default
            input_nodes_types = [Model.Input.INPUT_TYPE_DEFAULT]*len(input_nodes_names)

        graph_def = self.__import_graph(graph_pb_or_meta_path, session, out_node_names)
        with session.graph.as_default():
            inputs = []
            for name, shape, input_type in zip(input_nodes_names, input_nodes_shapes, input_nodes_types):
                self.__assert_node_in_graph(graph_def, name)
                input_tensor = session.graph.get_tensor_by_name(GraphHelper.indexed_tensor_name(name))

                batched_shape = []
                try:
                    tensor_shape = input_tensor.get_shape().as_list()
                    input_shape = list(map(int, shape.split(',')))
                    if len(input_shape) != len(tensor_shape):
                        raise ConverterError(code_to_message.get_error_message('ERROR_TF_INPUT_NODE_SHAPE_DIMS_MISMATCH'))
                    batched_shape = [1] * len(tensor_shape)
                    batched_shape[-len(input_shape):] = input_shape
                except ValueError:
                    pass

                if len(batched_shape) == 0:
                    try:
                        batched_shape = list(map(int, shape.split(',')))
                    except ValueError:
                        raise ConverterError(code_to_message.get_error_message('ERROR_TF_INVALID_INPUT_DIMS')(shape))

                inputs.append(Model.Input(name, batched_shape, input_type))

            visitable_graph = VisitableGraph(self.__get_graph_operations(graph_def, session.graph))
            visitable_graph.accept(GraphPrinter())

            return Model(graph_def, session, inputs, out_node_names)

    @classmethod
    def __get_graph_operations(cls, graph_def, graph):
        ops = [graph.get_operation_by_name(node.name) for node in graph_def.node]
        return ops

    @classmethod
    def __import_graph(cls, graph_path, session, out_nodes_names):
        """
        :type graph_path: str
        :type session: tensorflow.Session
        :type out_nodes_names: list[str]
        :rtype: tf.GraphDef
        """
        if not os.path.exists(graph_path):
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_GRAPH_FILE_DOES_NOT_EXIST')(graph_path))

        graph_path = os.path.abspath(graph_path)
        if graph_path.endswith('.meta'):
            checkpoint = graph_path.split('.meta')[0]
            graph_def = cls.__import_from_meta_graph(graph_path, checkpoint, out_nodes_names)
        else:
            graph_def = cls.__import_from_frozen_graph(graph_path)

        if len(graph_def.node) == 0:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_NODES_NOT_FOUND_IN_GRAPH'))

        with session.graph.as_default():
            tf.import_graph_def(graph_def, name="")
        return graph_def

    @classmethod
    def __import_from_frozen_graph(cls, graph_path):
        graph_def = tf.GraphDef()
        with open(graph_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    @classmethod
    def __import_from_meta_graph(cls, meta_graph_path, graph_path, out_nodes_names):
        """
        :type meta_graph_path: str
        :type graph_path: str
        :type out_nodes_names: list[str]
        :rtype: tensorflow.GraphDef
        """
        session = tf.Session(graph=tf.Graph())
        with session.graph.as_default():
            try:
                saver = tf.train.import_meta_graph(meta_graph_path)
            except AssertionError as e:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CANNOT_IMPORT_GRAPH_FROM_META')(e.message))

            if saver is None:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_GRAPH_META_EMPTY'))
            saver.restore(session, graph_path)

        graph_def = session.graph.as_graph_def(add_shapes=True)
        return cls.__freeze_graph(session, graph_def, out_nodes_names)

    @classmethod
    def __freeze_graph(cls, session, graph_def, out_nodes_names):
        for node_name in out_nodes_names:
            cls.__assert_node_in_graph(graph_def, node_name)
        frozen = graph_util.convert_variables_to_constants(session, graph_def, out_nodes_names)
        return frozen

    @classmethod
    def __assert_node_in_graph(cls, graph_def, node_name):
        if node_name not in [node.name for node in graph_def.node]:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_NODE_NOT_FOUND_IN_GRAPH')(node_name))


class Model(object):
    class Input(object):
        INPUT_TYPE_DEFAULT = "default"

        def __init__(self, name, shape, type):
            self.name = name  # str
            self.shape = shape  # list[int]
            self.type = type  # str

    def __init__(self, graph_def, session, inputs, out_nodes_names):
        """
        :type graph_def: tensorflow.GraphDef
        :type session: tensorflow.Session
        :type inputs: list[Model.Input]
        :type out_nodes_names: list[str]
        """
        self.graph_def = graph_def
        self.session = session
        self.inputs = inputs
        self.out_nodes_names = out_nodes_names
