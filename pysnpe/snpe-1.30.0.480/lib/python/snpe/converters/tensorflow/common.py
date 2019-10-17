#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from abc import ABCMeta, abstractmethod

from snpe.converters.common.utils import code_to_message
from .util import ConverterError


class LayerDescriptor(object):
    def __init__(self, layer_type, layer_name, operations, output_names=None):
        """
        Defines a base class to hold information regarding a layer's parameters extracted from a set of operations
        within a tensorflow.Graph.
        :type layer_type: str
        :type layer_name: str
        :type operations: list[tensorflow.Operation]
        """
        super(LayerDescriptor, self).__init__()
        self.layer_type = layer_type
        self.layer_name = str(layer_name)
        self.child_ops = operations
        self._ignored = False
        self._output_names = output_names

    @property
    def output_names(self):
        """
        :rtype: [str]
        """
        if self._output_names is None:
            return [str(self.child_ops[-1].outputs[0].name)]
        else:
            return self._output_names

    @output_names.setter
    def output_names(self, value):
        self._output_names = value

    def get_output_names_for(self, input_tensors):
        """
        :type input_tensors: [tensorflow.Tensor]
        :rtype: [str]
        """
        output_tensors = [t for o in self.child_ops for t in o.outputs]
        return [str(t.name) for t in input_tensors if t in output_tensors]

    @classmethod
    def none(cls):
        """
        Returns a LayerDescriptor to represent an invalid layer. This is used
        by implementations of LayerBuilder.build_layer_descriptor(..) to convey
        not being able to extract a layer from a set of graph operations.
        :rtype: LayerDescriptor.
        """
        return LayerDescriptor("none", "none", [])

    def set_ignored(self, ignored):
        """
        Sets the descriptor as ignored which will cause
        the conversion process to not build a layer from
        this descriptor.
        :type ignored: bool
        :return:
        """
        self._ignored = ignored

    @property
    def is_ignored(self):
        return self._ignored

    def is_output_op(self, op):
        if self._output_names is not None:
            return op.outputs[0].name in self._output_names
        else:
            return self.child_ops[-1] == op

    def is_input_op(self, op):
        return op in self.child_ops

    def is_input_tensor(self, op, tensor):
        return True

    def __eq__(self, other):
        result = False
        if isinstance(other, self.__class__):
            result = (self.layer_name == other.layer_name)
            result = result and (self.child_ops == other.child_ops)
        return result

    def __hash__(self):
        return hash(tuple(self.layer_name, ) + tuple(self.child_ops))


class InputLayerDescriptor(LayerDescriptor):
    def __init__(self, name, nodes):
        super(InputLayerDescriptor, self).__init__('Input', name, nodes)


class LayerResolver(object):
    """
    Defines an API for each type of layer to extend and implement it's layer parameter resolution.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def resolve_layer(self, graph_matcher, graph_helper):
        """
        :type graph_matcher: GraphMatcher
        :type graph_helper: GraphHelper
        :rtype: list(LayerDescriptor)
        """
        pass

    def is_final_resolution(self):
        return False


class LayerBuilder(object):
    """
    Defines an API for each type of layer to extend and implement how to build the layer in the target format.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        Creates  a layer from the specified LayerDescriptor and returns an int representing the layer unique id.
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: converters.tensorflow.common.LayerDescriptor
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :rtype: int
        """
        pass

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        Allows builders to fuse or skip layers.
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: converters.tensorflow.common.LayerDescriptor
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        """
        pass

    @classmethod
    def get_input_name(cls, converter_context, descriptor, input_descriptors):
        """
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type input_descriptors: [LayerDescriptor]
        :type descriptor: LayerDescriptor
        :rtype: str
        """
        if len(input_descriptors) > 1:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_LAYER_INPUT_COUNT_ERROR')(
                input_descriptors[0].layer_type, 1, len(input_descriptors)
            ))

        input_names = cls.get_input_names(converter_context, descriptor, input_descriptors)
        if len(input_names) == 0:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_LAYER_NO_INPUT_FOUND')(
                descriptor.layer_type, descriptor.layer_name))

        if len(input_names) > 1:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_LAYER_INPUT_COUNT_ERROR')(
                input_descriptors[0].layer_type, 1, len(input_descriptors)
            ))

        return input_names[0]

    @classmethod
    def get_input_names(cls, converter_context, descriptor, input_descriptors):
        """
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type input_descriptors: [LayerDescriptor]
        :type descriptor: LayerDescriptor
        :rtype: str
        """
        input_names = []
        for d in input_descriptors:
            input_tensors = converter_context.get_output_tensors_between(d, descriptor)
            input_names.extend(d.get_output_names_for(input_tensors))

        if len(input_names) == 0:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_LAYER_NO_INPUT_FOUND')(
                descriptor.layer_type, descriptor.layer_name))
        return input_names
