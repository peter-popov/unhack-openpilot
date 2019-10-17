#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.util import ConverterError
from abc import ABCMeta
from abc import abstractmethod
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class EltWiseUnaryLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    def __init__(self, layer_type, op_type, descriptor_class):
        super(EltWiseUnaryLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            NonConsumableConverterSequenceNode('input1', ['?']),
        ])
        self.sequence.set_inputs('root', ['input1'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        non_const_input_sequences = [self.sequence]
        for sequence in non_const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes)
                descriptors.append(descriptor)

        return descriptors


class EltWiseUnaryLayerBuilder(LayerBuilder):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        pass

class EltWiseUnarySqrtLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnarySqrtLayerResolver, self).__init__('ElementWiseUnarySqrt', 'Sqrt', EltWiseUnarySqrtLayerResolver.Descriptor)


class EltWiseUnarySqrtLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseUnarySqrtLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_unary_sqrt_layer(descriptor.layer_name,
                                                                        input_name,output_name)
