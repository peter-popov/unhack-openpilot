#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from snpe.converters.tensorflow.util import ConverterError


class PermuteLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, order, output_names=None):
            super(PermuteLayerResolver.Descriptor, self).__init__('Permute', name, nodes, output_names=output_names)
            self.order = order

    def __init__(self):
        self.sequence_with_explicit_order = GraphSequence([
            ConverterSequenceNode('root', ['Transpose']),
            ConverterSequenceNode('order', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence_with_explicit_order.set_inputs('root', ['input', 'order'])
        self.sequence_with_explicit_order.set_outputs(['root'])

        self.sequence_with_implicit_order = GraphSequence([
            ConverterSequenceNode('root', ['Transpose']),
            ConverterSequenceNode('order', ['Sub']),
            ConverterSequenceNode('a', ['Sub']),
            ConverterSequenceNode('b', ['Const']),
            ConverterSequenceNode('c', ['Range']),
            ConverterSequenceNode('d', ['Const']),
            ConverterSequenceNode('e', ['Const']),
            ConverterSequenceNode('f', ['Rank']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])

        self.sequence_with_implicit_order.set_inputs('root', ['input', 'order'])
        self.sequence_with_implicit_order.set_inputs('order', ['a', 'c'])
        self.sequence_with_implicit_order.set_inputs('a', ['b', 'f'])
        self.sequence_with_implicit_order.set_inputs('c', ['d', 'e', 'f'])
        self.sequence_with_implicit_order.set_inputs('f', ['input'])
        self.sequence_with_implicit_order.set_outputs(['root'])

        self.sequences = [self.sequence_with_explicit_order, self.sequence_with_implicit_order]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                transpose_op = match['root']
                input_op = match['input']
                order_op = match['order']

                order_tensor = graph_helper.evaluate_tensor_output(order_op.outputs[0])

                input_shape = graph_helper.get_op_output_shape(input_op)
                order_shape = graph_helper.get_op_output_shape(order_op)

                input_rank = len(input_shape)
                order_rank = len(order_shape)
                try:
                    assert order_rank == 1
                    for d in range(input_rank):
                        assert d in order_tensor
                except AssertionError:
                    raise ConverterError(code_to_message.get_error_message(
                        'ERROR_TF_PERMUTE_INVALID_ORDER_TENSOR')(str(order_tensor)))

                consumed_nodes = match.consumed_nodes
                permute_descriptor = PermuteLayerResolver.Descriptor(
                    str(transpose_op.name), consumed_nodes, order_tensor,
                    output_names=[str(transpose_op.outputs[0].name)])
                descriptors.extend([permute_descriptor])

        return descriptors


class PermuteLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PermuteLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_permute_layer(name=descriptor.layer_name,
                                                         order=descriptor.order.tolist(),
                                                         input_name=input_name,
                                                         output_name=output_name)
