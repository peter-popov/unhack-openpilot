#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import math
from abc import ABCMeta
import sys

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.util import scoped_op_name
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class PoolingLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, operations, pooling_type, strides, padding, kernel_dims):
            super(PoolingLayerResolver.Descriptor, self).__init__(layer_type, name, operations)
            self.pooling_type = pooling_type
            self.strides = strides
            self.padding = padding
            self.kernel_dims = kernel_dims

    def __init__(self, layer_type, descriptor_type, pooling_type, op_type):
        super(PoolingLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._descriptor_type = descriptor_type
        self._polling_type = pooling_type
        self._op_type = op_type

        self.sequence = GraphSequence([ConverterSequenceNode('root', [self._op_type])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            pooling_op = match['root']
            kernel_dims = pooling_op.get_attr('ksize')
            strides = pooling_op.get_attr('strides')
            padding = pooling_op.get_attr('padding')
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                self._descriptor_type(self._layer_type, str(pooling_op.name), consumed_nodes,
                                      self._polling_type,
                                      strides, padding, kernel_dims))
        return potential_descriptors


class PoolingLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PoolingLayerResolver.Descriptor
        :rtype: int
        """
        input_dims = converter_context.get_input_layer_output_shape_for(descriptor.child_ops[0])

        pad_y, pad_x, padding_strategy = self.calculate_padding(descriptor.padding, input_dims[1:3],
                                              descriptor.strides[1:3], descriptor.kernel_dims[1:3])

        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_pooling_layer(name=descriptor.layer_name,
                                                         pool_type=descriptor.pooling_type,
                                                         pool_size_x=descriptor.kernel_dims[2],
                                                         pool_size_y=descriptor.kernel_dims[1],
                                                         pool_stride_x=descriptor.strides[2],
                                                         pool_stride_y=descriptor.strides[1],
                                                         pad_x=pad_x,
                                                         pad_y=pad_y,
                                                         padding_size_strategy=padding_strategy,
                                                         input_name=input_name,
                                                         output_name=output_name,
                                                         pool_region_include_padding=False)

    @classmethod
    def calculate_padding(cls, padding_type, input_size, strides, pool_dims):
        pad_y, pad_x = 0, 0
        padding_size_strategy = modeltools.PADDING_SIZE_IMPLICIT_VALID
        if padding_type.decode() == 'SAME':
            output_height = math.ceil(float(input_size[0]) / float(strides[0]))
            output_width = math.ceil(float(input_size[1]) / float(strides[1]))
            pad_y = ((output_height - 1) * strides[0] + pool_dims[0] - input_size[0])
            pad_x = ((output_width - 1) * strides[1] + pool_dims[1] - input_size[1])
            # We divide by two and truncate if odd padding given the runtime will
            # take care of Asymmetry
            pad_y /= 2
            pad_x /= 2
            padding_size_strategy = modeltools.PADDING_SIZE_IMPLICIT_SAME
        return int(pad_y), int(pad_x), padding_size_strategy


class AvgPoolingLayerResolver(PoolingLayerResolver):
    class Descriptor(PoolingLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(AvgPoolingLayerResolver, self).__init__('AvgPooling', AvgPoolingLayerResolver.Descriptor,
                                                      modeltools.POOL_AVG, 'AvgPool')


class MaxPoolingLayerResolver(PoolingLayerResolver):
    class Descriptor(PoolingLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(MaxPoolingLayerResolver, self).__init__('MaxPooling', MaxPoolingLayerResolver.Descriptor,
                                                      modeltools.POOL_MAX, 'MaxPool')
