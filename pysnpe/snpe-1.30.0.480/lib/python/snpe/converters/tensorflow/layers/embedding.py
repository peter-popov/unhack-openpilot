#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)
from snpe.converters.common.utils.code_to_message import get_error_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerBuilder, LayerResolver, ConverterError
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from snpe.converters.tensorflow.layers.constant import ConstantLayerResolver
from snpe.converters.tensorflow.util import GraphHelper


class EmbeddingLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, output_dim, input_names):
            super(EmbeddingLayerResolver.Descriptor, self).__init__('Embedding', name, nodes)
            self.output_dim = output_dim
            # There will be two input descriptors for embedding, ids and params.
            # Hold input names ordered list as CPU runtime needs 'ids' name comes
            # firstly, 'params' comes secondly
            self.input_names = input_names

    def __init__(self):

        sequence_1 = GraphSequence([
            ConverterSequenceNode('gather', ['GatherV2']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('axis', ['?']),
            NonConsumableConverterSequenceNode('indices', ['Placeholder'])
        ])
        sequence_1.set_inputs('gather', ['params', 'axis', 'indices'])
        sequence_1.set_outputs(['gather'])

        # Filter seqs 2
        sequence_2 = GraphSequence([
            ConverterSequenceNode('gather', ['Gather']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('indices', ['Placeholder'])
        ])
        sequence_2.set_inputs('gather', ['params', 'indices'])
        sequence_2.set_outputs(['gather'])

        self.sequences = [sequence_1, sequence_2]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                embedding_lookup_op = match['gather']
                consumed_nodes = match.consumed_nodes
                output_dim = graph_helper.get_op_output_shape(embedding_lookup_op)

                # get rid of axis op from inputs
                inputs_sanitized = []
                if len(embedding_lookup_op.inputs) == 2:
                    inputs_sanitized.extend(embedding_lookup_op.inputs)
                elif len(embedding_lookup_op.inputs) == 3:
                    for tensor in embedding_lookup_op.inputs:
                        # exclude axis op
                        if tensor.op.type == 'Const' and len(graph_helper.get_op_output_shape(tensor.op)) == 0:
                            continue
                        else:
                            inputs_sanitized.append(tensor)

                # take ids always as input, params as input or not. So ids tensor type is Placeholder
                if all(tensor.op.type == 'Placeholder' for tensor in inputs_sanitized):
                    ids_candidate, params_candidate = inputs_sanitized[0], inputs_sanitized[1]
                    ids_candidate_shape = graph_helper.get_op_output_shape(ids_candidate.op)
                    params_candidate_shape = graph_helper.get_op_output_shape(params_candidate.op)
                    # Do shape check to determine which are ids and params.
                    # Make assumption that ids shape comes firstly in output dim.
                    # If they have the same shape, then we've got the only way to
                    # determine ids and params by checking name. Otherwise raise error.
                    if ids_candidate_shape == params_candidate_shape:
                        ids_candidate = [tensor for tensor in inputs_sanitized if tensor.name.find("ids") != -1]
                        params_candidate = [tensor for tensor in inputs_sanitized if tensor.name.find("params") != -1]
                        if len(ids_candidate) == 0 or len(params_candidate) == 0:
                            raise ConverterError(get_error_message('ERROR_TF_EMBEDDING_CANNOT_RESOLVE_PARAMS_AND_IDS'))
                    else:
                        if output_dim[:len(ids_candidate_shape)] != ids_candidate_shape and \
                                output_dim[:len(params_candidate_shape)] == params_candidate_shape:
                            ids_candidate, params_candidate = params_candidate, ids_candidate
                    descriptors.append(EmbeddingLayerResolver.Descriptor(str(embedding_lookup_op.name),
                        consumed_nodes, output_dim, [str(ids_candidate.name), str(params_candidate.name)]))
                else:
                    ids = [tensor for tensor in inputs_sanitized if tensor.op.type == 'Placeholder'][0]
                    params_candidate_op = [tensor.op for tensor in inputs_sanitized if tensor.op.type != 'Placeholder'][0]

                    const_consumed_ops = [params_candidate_op]
                    while params_candidate_op.type == 'Identity':
                        params_candidate_op = params_candidate_op.inputs[0].op
                        const_consumed_ops.append(params_candidate_op)

                    embedding_descriptor = EmbeddingLayerResolver.Descriptor(str(embedding_lookup_op.name),
                                                                             consumed_nodes, output_dim,
                                                                             [str(ids.name),
                                                                              GraphHelper.indexed_tensor_name(
                                                                                  params_candidate_op.name)])
                    descriptors.append(embedding_descriptor)
                    if params_candidate_op.type == 'Const':
                        embedding_shape = graph_helper.get_op_output_shape(params_candidate_op)
                        const_tensor = graph_helper.evaluate_tensor_output(params_candidate_op.outputs[0])
                        const_descriptor = ConstantLayerResolver.Descriptor(str(params_candidate_op.name), const_consumed_ops,
                                                                            const_tensor, embedding_shape,
                                                                            embedding_descriptor)
                        descriptors.append(const_descriptor)
        return descriptors


class EmbeddingLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EmbeddingLayerResolver.Descriptor
        :rtype: int
        """
        if len(input_descriptors) != 2:
            raise ConverterError(get_error_message('ERROR_TF_EMBEDDING_REQUIRES_2_INPUTS'))

        output_name = descriptor.output_names[0]
        return converter_context.model.add_embedding_layer(name=descriptor.layer_name,
                                                           output_dim=descriptor.output_dim,
                                                           input_names=descriptor.input_names,
                                                           output_name=output_name,
                                                           partition_strategy=modeltools.EMBEDDING_PARTITION_STRATEGY_MOD)


