#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from snpe.converters.tensorflow.util import ConverterError


class FullyConnectedLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, matmul_op, bias_op, weights, biases, output_names=None):
            super(FullyConnectedLayerResolver.Descriptor, self).__init__('FullyConnected', name, nodes, output_names=output_names)
            self.matmul_op = matmul_op
            self.bias_op = bias_op
            self.weights = weights
            self.biases = biases

    def __init__(self):

        sequence =  GraphSequence([
            ConverterSequenceNode('matmul_op', ['MatMul']),
            ConverterSequenceNode('bias_op', ['BiasAdd', 'Add']),  # output
            NonConsumableConverterSequenceNode('biases', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('weights', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence.set_inputs('matmul_op', ['inputs', 'weights'])
        sequence.set_inputs('bias_op', ['matmul_op', 'biases'])
        sequence.set_outputs(['bias_op'])

        sequence_without_bias = GraphSequence([
            ConverterSequenceNode('matmul_op', ['MatMul']),
            NonConsumableConverterSequenceNode('weights', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_without_bias.set_inputs('matmul_op', ['inputs', 'weights'])
        sequence_without_bias.set_outputs(['matmul_op'])

        self.sequences = [sequence_without_bias,sequence]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                matmul_op = match['matmul_op']
                weights_op = match['weights']
                biases_op = None
                bias_add_op = None
                if weights_op.type not in ['Identity', 'Const', 'FakeQuantWithMinMaxVars']:
                    raise ConverterError(code_to_message.get_error_message('ERROR_TF_MATMUL_RESOLVE_WEIGHTS')(matmul_op.name))
                weights = graph_helper.evaluate_tensor_output(weights_op.outputs[0])
                try:
                    bias_add_op = match['bias_op']
                    biases_op = match['biases']
                except KeyError:
                    pass
                if biases_op is not None and bias_add_op is not None:
                    if biases_op.type not in ['Identity', 'Const']:
                        # do we still need this check ?
                        raise ConverterError(
                            code_to_message.get_error_message('ERROR_TF_MATMUL_RESOLVE_BIAS')(bias_add_op.name))
                    biases = graph_helper.evaluate_tensor_output(biases_op.outputs[0])
                else:
                    biases = np.zeros(weights.shape[-1], dtype=np.float32)

                consumed_nodes = match.consumed_nodes
                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in sequence.output_nodes]
                descriptors.append(
                    FullyConnectedLayerResolver.Descriptor(str(matmul_op.name), consumed_nodes,
                                                       matmul_op, bias_add_op, weights, biases,
                                                       output_names=output_op_nodes_names))
        return descriptors


class FullyConnectedLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FullyConnectedLayerResolver.Descriptor
        :rtype: int
        """
        weight_tensor = np.transpose(descriptor.weights, (1, 0)).copy()

        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_fc_layer(name=descriptor.layer_name,
                                                    weights_list=[weight_tensor],
                                                    bias=descriptor.biases,
                                                    input_names=input_names,
                                                    output_name=output_name)
