#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from snpe.converters.tensorflow.util import ConverterError


class PReLuLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, coefficients, output_names):
            super(PReLuLayerResolver.Descriptor, self).__init__('PReLU', name, operations,
                                                                output_names=output_names)
            self.coefficients = coefficients

    def __init__(self):
        sequence_prelu = GraphSequence([
            ConverterSequenceNode('a', ['Relu']),
            ConverterSequenceNode('b', ['Abs']),
            ConverterSequenceNode('c', ['Sub']),
            ConverterSequenceNode('d', ['Mul']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('f', ['Add']),  # output
            ConverterSequenceNode('unknown', ['?']),
            ConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_prelu.set_inputs('a', ['inputs'])
        sequence_prelu.set_inputs('b', ['inputs'])
        sequence_prelu.set_inputs('c', ['inputs', 'b'])
        sequence_prelu.set_inputs('d', ['alphas', 'c'])
        sequence_prelu.set_inputs('e', ['d', 'unknown'])
        sequence_prelu.set_inputs('f', ['a', 'e'])
        sequence_prelu.set_outputs(['f'])

        sequence_prelu_negative_alpha = GraphSequence([
            ConverterSequenceNode('a', ['Relu']),
            ConverterSequenceNode('b', ['Neg']),
            ConverterSequenceNode('c', ['Neg']),
            ConverterSequenceNode('d', ['Relu']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('f', ['Add']),  # output
            ConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_prelu_negative_alpha.set_inputs('a', ['inputs'])
        sequence_prelu_negative_alpha.set_inputs('b', ['inputs'])
        sequence_prelu_negative_alpha.set_inputs('c', ['alphas'])
        sequence_prelu_negative_alpha.set_inputs('d', ['b'])
        sequence_prelu_negative_alpha.set_inputs('e', ['d', 'c'])
        sequence_prelu_negative_alpha.set_inputs('f', ['a', 'e'])
        sequence_prelu_negative_alpha.set_outputs(['f'])

        sequence_prelu_negative_relu = GraphSequence([
            ConverterSequenceNode('relu_pos', ['Relu']),
            ConverterSequenceNode('neg_1', ['Neg']),
            ConverterSequenceNode('neg_2', ['Neg']),
            ConverterSequenceNode('relu_neg', ['Relu']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('f', ['Add']),  # output
            ConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_prelu_negative_relu.set_inputs('relu_pos', ['inputs'])
        sequence_prelu_negative_relu.set_inputs('neg_1', ['inputs'])
        sequence_prelu_negative_relu.set_inputs('relu_neg', ['neg_1'])
        sequence_prelu_negative_relu.set_inputs('neg_2', ['relu_neg'])
        sequence_prelu_negative_relu.set_inputs('mul', ['neg_2', 'alphas'])
        sequence_prelu_negative_relu.set_inputs('f', ['relu_pos', 'mul'])
        sequence_prelu_negative_relu.set_outputs(['f'])

        self.sequences = [sequence_prelu, sequence_prelu_negative_alpha, sequence_prelu_negative_relu]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                coefficients = match['alphas']
                add_op = match['f']
                if coefficients.type not in ['Identity', 'Const']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_RESOLVE_PRELU_COEFF'))

                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in sequence.output_nodes]
                consumed_nodes = match.consumed_nodes
                potential_descriptors.append(
                    PReLuLayerResolver.Descriptor(str(add_op.name), consumed_nodes,
                                                  graph_helper.evaluate_tensor_output(coefficients.outputs[0]),
                                                  output_names=output_op_nodes_names))
        return potential_descriptors


class PReLuLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReluLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_prelu_layer(name=descriptor.layer_name,
                                                       coeff=descriptor.coefficients.flatten().tolist(),
                                                       input_name=input_name,
                                                       output_name=output_name)
