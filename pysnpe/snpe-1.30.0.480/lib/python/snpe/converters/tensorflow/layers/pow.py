#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from snpe.converters.tensorflow.util import ConverterError


class PowLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, scale, shift, power, output_names=None):
            super(PowLayerResolver.Descriptor, self).__init__('Pow', name, nodes, output_names=output_names)
            self.scale = scale
            self.power = power
            self.shift = shift

    def __init__(self):
        sequence_scalar_pow = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('pow', ['Pow']),
            ConverterSequenceNode('const', ['Const'])
        ])
        sequence_scalar_pow.set_inputs('pow', ['input', 'const'])
        sequence_scalar_pow.set_outputs(['pow'])

        self.sequences = [sequence_scalar_pow]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                pow_op = match['pow']
                const_values_op = match['const']
                const_values = graph_helper.evaluate_tensor_output(const_values_op.outputs[0])

                # only scalar power op is supported
                if not np.isscalar(const_values):
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_POW_CONSTANT_NOT_SCALAR'))

                consumed_nodes = match.consumed_nodes
                pow_descriptor = PowLayerResolver.Descriptor(
                    str(pow_op.name), consumed_nodes, 1, 0, const_values,
                    output_names=[str(pow_op.outputs[0].name)])
                descriptors.extend([pow_descriptor])

        return descriptors


class PowLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PowLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_power_layer(name=descriptor.layer_name,
                                                     input_name=input_name,
                                                     scale=descriptor.scale,
                                                     shift=descriptor.shift,
                                                     power=descriptor.power,
                                                     output_name=output_name)
