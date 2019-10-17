#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from snpe.converters.tensorflow.util import ConverterError


class ConstantLayerResolver(LayerResolver, object):
    def resolve_layer(self, graph_matcher, graph_helper):
        raise ConverterError('Constant layers are resolved by other resolvers!')

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, value, shape, consumer):
            super(ConstantLayerResolver.Descriptor, self).__init__('Constant', name, nodes)
            self.value = value
            self.shape = shape
            self.consumer = consumer

        def is_input_tensor(self, op, tensor):
            return False

class ConstantLayerBuilder(LayerBuilder):

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        ignored = [d for d in output_descriptors if isinstance(d, IgnoredLayersResolver.Descriptor)]
        if ignored == output_descriptors:
            descriptor.set_ignored(True)

        if len(output_descriptors) == 1 and not descriptor.consumer == output_descriptors[0]:
            descriptor.set_ignored(True)

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FillLayerResolver.Descriptor
        :rtype: int
        """
        if not isinstance(descriptor.value, np.ndarray):
            array = np.zeros(descriptor.shape, dtype=np.float32)
            array[...] = descriptor.value
            descriptor.value = array
        return converter_context.model.add_const_layer(descriptor.output_names[0],
                                                       list(descriptor.shape),
                                                       descriptor.value,
                                                       True)
