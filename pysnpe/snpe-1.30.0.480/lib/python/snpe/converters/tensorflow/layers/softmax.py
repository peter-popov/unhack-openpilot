#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class SoftmaxLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        sequence_two_dim_softmax = GraphSequence([ConverterSequenceNode('root', ['SoftMax'])])
        sequence_two_dim_softmax.set_outputs(['root'])

        sequence_multi_dim_softmax = GraphSequence([
            ConverterSequenceNode('max', ['Max']),
            ConverterSequenceNode('max_reduction_indicies', ['Const']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('exp', ['Exp']),
            ConverterSequenceNode('sum', ['Sum']),
            ConverterSequenceNode('sum_reduction_indicies', ['Const']),
            ConverterSequenceNode('root', ['RealDiv']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        sequence_multi_dim_softmax.set_inputs('max', ['input', 'max_reduction_indicies'])
        sequence_multi_dim_softmax.set_inputs('sub', ['input', 'max'])
        sequence_multi_dim_softmax.set_inputs('exp', ['sub'])
        sequence_multi_dim_softmax.set_inputs('sum', ['exp', 'sum_reduction_indicies'])
        sequence_multi_dim_softmax.set_inputs('root', ['exp', 'sum'])
        sequence_multi_dim_softmax.set_outputs(['root'])

        self.sequences = [sequence_two_dim_softmax, sequence_multi_dim_softmax]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                softmax_op = match['root']
                consumed_nodes = match.consumed_nodes
                potential_descriptors.append(
                    SoftmaxLayerResolver.Descriptor('Softmax', str(softmax_op.name), consumed_nodes))
        return potential_descriptors


class SoftmaxLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: SoftmaxLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_softmax_layer(name=descriptor.layer_name,
                                                         input_name=input_name,
                                                         output_name=output_name)
