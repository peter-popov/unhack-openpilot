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


class BatchNormLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, bn_mul_op, pre_calculated=False, *args, **kwargs):
            super(BatchNormLayerResolver.Descriptor, self).__init__('BatchNormalization', name, operations,
                                                                    output_names=kwargs.get('output_names', None))
            self.bn_mul_op = bn_mul_op
            self.pre_calculated = pre_calculated
            if self.pre_calculated:
                self.weights = kwargs.get('weights')
                self.biases = kwargs.get('biases')
            else:
                mean = kwargs.get('mean')
                variance = kwargs.get('variance')
                epsilon = kwargs.get('epsilon')
                self.scale = kwargs.get('scale')
                self.beta = kwargs.get('beta')
                stddev = 1 / np.sqrt(variance + epsilon)
                scaled_stddev = stddev * self.scale
                scaled_variance = variance * scaled_stddev
                scaled_mean = mean * scaled_stddev
                self.weights = scaled_stddev
                self.biases = (-1 * scaled_mean) + self.beta

    def resolve_layer(self, graph_matcher, graph_helper):
        raise ConverterError(code_to_message.get_error_message('ERROR_TF_GENERAL_ABSTRACT_CLASS_MUST_BE_INHERITED'))


class ScaledBatchNormLayerResolver(BatchNormLayerResolver):

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('a', ['Add']),
            ConverterSequenceNode('b', ['Rsqrt']),
            ConverterSequenceNode('c', ['Mul']),
            ConverterSequenceNode('d', ['Mul']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('f', ['Sub']),
            ConverterSequenceNode('g', ['Add']),
            ConverterSequenceNode('scale', ['?']),
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('mean', ['?']),
            ConverterSequenceNode('beta', ['?']),
            ConverterSequenceNode('variance', ['?']),
            ConverterSequenceNode('epsilon', ['?'])
        ])
        self.sequence.set_inputs('a', ['variance', 'epsilon'])
        self.sequence.set_inputs('b', ['a'])
        self.sequence.set_inputs('c', ['b', 'scale'])
        self.sequence.set_inputs('d', ['c', 'input'])
        self.sequence.set_inputs('e', ['c', 'mean'])
        self.sequence.set_inputs('f', ['e', 'beta'])
        self.sequence.set_inputs('g', ['d', 'f'])
        self.sequence.set_outputs(['g'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            variance_op = match['variance']
            epsilon_op = match['epsilon']
            if variance_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_VARIANCE'))
            variance = graph_helper.evaluate_tensor_output(variance_op.outputs[0])

            if epsilon_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_EPSILON'))
            epsilon = graph_helper.evaluate_tensor_output(epsilon_op.outputs[0])

            scale_op = match['scale']
            if scale_op.type not in ['Identity', 'Const', 'Fill']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_SCALE'))
            scale = graph_helper.evaluate_tensor_output(scale_op.outputs[0])

            mean_op = match['mean']
            if mean_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_MEAN'))
            mean = graph_helper.evaluate_tensor_output(mean_op.outputs[0])

            beta_op = match['beta']
            if beta_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_BETA'))
            beta = graph_helper.evaluate_tensor_output(beta_op.outputs[0])

            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]
            descriptors.append(
                BatchNormLayerResolver.Descriptor(str(match['d'].name),
                                                  match.consumed_nodes,
                                                  bn_mul_op=match['d'],
                                                  mean=mean,
                                                  variance=variance,
                                                  epsilon=epsilon,
                                                  scale=scale,
                                                  beta=beta,
                                                  output_names=output_op_nodes_names))
        return descriptors


class UnscaledBatchNormLayerResolver(BatchNormLayerResolver):

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('a', ['Add']),
            ConverterSequenceNode('b', ['Rsqrt']),
            ConverterSequenceNode('c', ['Mul']),
            ConverterSequenceNode('d', ['Mul']),
            ConverterSequenceNode('e', ['Sub']),
            ConverterSequenceNode('f', ['Add']),  # output
            NonConsumableConverterSequenceNode('inputs', ['?']),
            ConverterSequenceNode('mean', ['?']),
            ConverterSequenceNode('beta', ['?']),
            ConverterSequenceNode('variance', ['?']),
            ConverterSequenceNode('epsilon', ['?'])
        ])
        self.sequence.set_inputs('a', ['variance', 'epsilon'])
        self.sequence.set_inputs('b', ['a'])
        self.sequence.set_inputs('c', ['b', 'inputs'])
        self.sequence.set_inputs('d', ['b', 'mean'])
        self.sequence.set_inputs('e', ['d', 'beta'])
        self.sequence.set_inputs('f', ['c', 'e'])
        self.sequence.set_outputs(['f'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            variance_op = match['variance']
            epsilon_op = match['epsilon']
            if variance_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_VARIANCE'))
            variance = graph_helper.evaluate_tensor_output(variance_op.outputs[0])

            if epsilon_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_EPSILON'))
            epsilon = graph_helper.evaluate_tensor_output(epsilon_op.outputs[0])

            mean_op = match['mean']
            if mean_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_MEAN'))
            mean = graph_helper.evaluate_tensor_output(mean_op.outputs[0])

            beta_op = match['beta']
            if beta_op.type not in ['Identity', 'Const']:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_RESOLVE_BETA'))
            beta = graph_helper.evaluate_tensor_output(beta_op.outputs[0])
            scale = np.ones(shape=mean.shape, dtype=np.float32)
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]
            potential_descriptors.append(
                BatchNormLayerResolver.Descriptor(str(match['c'].name),
                                                  consumed_nodes,
                                                  bn_mul_op=match['c'],
                                                  mean=mean,
                                                  variance=variance,
                                                  epsilon=epsilon,
                                                  scale=scale,
                                                  beta=beta,
                                                  output_names=output_op_nodes_names))
        return potential_descriptors


class GenericBatchNormLayerResolver(BatchNormLayerResolver):
    class Descriptor(BatchNormLayerResolver.Descriptor):
        pass

    def __init__(self):
        self.sequence = GraphSequence([
            NonConsumableConverterSequenceNode('inputs', ['?']),
            ConverterSequenceNode('a', ['Mul']),
            ConverterSequenceNode('b', ['Add']),
            ConverterSequenceNode('weights', ['Const', 'Identity']),
            ConverterSequenceNode('biases', ['Const', 'Identity'])
        ])
        self.sequence.set_inputs('a', ['inputs', 'weights'])
        self.sequence.set_inputs('b', ['a', 'biases'])
        self.sequence.set_outputs(['b'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            inputs_op = match['inputs']
            biases_op = match['biases']
            weights_op = match['weights']

            inputs_shape = graph_helper.get_op_output_shape(inputs_op)
            biases_op = graph_helper.evaluate_tensor_output(biases_op.outputs[0])
            weights_op = graph_helper.evaluate_tensor_output(weights_op.outputs[0])

            if np.isscalar(biases_op):
                biases_op = self._broadcast_tensor(biases_op, inputs_shape)
            if np.isscalar(weights_op):
                weights_op = self._broadcast_tensor(weights_op, inputs_shape)

            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]
            bn_op = match['a']
            potential_descriptors.append(
                GenericBatchNormLayerResolver.Descriptor(str(bn_op.name),
                                                         consumed_nodes,
                                                         bn_mul_op=bn_op,
                                                         pre_calculated=True,
                                                         weights=weights_op,
                                                         biases=biases_op,
                                                         output_names=output_op_nodes_names))
        return potential_descriptors

    @classmethod
    def _broadcast_tensor(cls, tensor, shape):
        broadcasted_tensor = np.zeros(shape, dtype=np.float32)
        broadcasted_tensor = broadcasted_tensor + tensor
        return broadcasted_tensor


class BatchNormWithGlobalNormLayerResolver(BatchNormLayerResolver):
    class Descriptor(BatchNormLayerResolver.Descriptor):
        pass

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['BatchNormWithGlobalNormalization'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            bn_op = match['root']
            parameter_tensors = self._const_inputs(graph_helper, bn_op)
            if len(parameter_tensors) < 4:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_GLOBALNORMALIZATION_INPUT'))
            epsilon = bn_op.get_attr('variance_epsilon')
            mean = parameter_tensors[0]
            variance = parameter_tensors[1]
            beta = parameter_tensors[2]
            scale = parameter_tensors[3]
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                BatchNormWithGlobalNormLayerResolver.Descriptor(str(bn_op.name),
                                                                consumed_nodes,
                                                                bn_mul_op=bn_op,
                                                                mean=mean,
                                                                variance=variance,
                                                                epsilon=epsilon,
                                                                scale=scale,
                                                                beta=beta))
        return potential_descriptors

    @classmethod
    def _const_inputs(cls, graph_helper, bn_op):
        return [graph_helper.evaluate_tensor_output(tensor) for tensor in bn_op.inputs if tensor.op.type == 'Const']


class FusedBatchNormNormLayerResolver(BatchNormLayerResolver):
    class Descriptor(BatchNormLayerResolver.Descriptor):
        pass

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['FusedBatchNorm'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        potential_descriptors = []
        for match in matches:
            bn_op = match['root']
            parameter_tensors = self._get_parameter_tensors(graph_helper, bn_op)
            if len(parameter_tensors) < 4:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_BATCHNORM_GLOBALNORMALIZATION_INPUT'))
            epsilon = bn_op.get_attr('epsilon')
            # we want the last 4 inputs, as sometimes non-parameter input can be of type of Identity(eg: seen in
            # mobilenet fpn ssd)
            scale = parameter_tensors[-4]
            beta = parameter_tensors[-3]
            mean = parameter_tensors[-2]
            variance = parameter_tensors[-1]
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                FusedBatchNormNormLayerResolver.Descriptor(str(bn_op.name),
                                                           consumed_nodes,
                                                           bn_mul_op=bn_op,
                                                           mean=mean,
                                                           variance=variance,
                                                           epsilon=epsilon,
                                                           scale=scale,
                                                           beta=beta))
        return potential_descriptors

    @classmethod
    def _get_parameter_tensors(cls, graph_helper, bn_op):
        parameter_tensors = [t for t in bn_op.inputs if t.op.type in ['Const', 'Identity']]
        tensors_outputs = graph_helper.evaluate_tensors_output(parameter_tensors)
        return [tensors_outputs[t] for t in parameter_tensors]


class BatchNormLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: BatchNormLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        return converter_context.model.add_batchnorm_layer(descriptor.layer_name,
                                                           descriptor.weights,
                                                           descriptor.biases,
                                                           compute_statistics=False,
                                                           use_mu_sigma=False,
                                                           across_spatial=False,
                                                           input_name=input_name,
                                                           output_name=descriptor.output_names[0])


