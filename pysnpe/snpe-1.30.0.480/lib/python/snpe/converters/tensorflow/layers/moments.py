#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import numpy as np
from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.util import ConverterError
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class MomentsLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axes, keep_dims, output_names=None):
            super(MomentsLayerResolver.Descriptor, self).__init__('Moments', name, nodes, output_names=output_names)
            self.axes = axes
            self.keep_dims = keep_dims

    def __init__(self):
        super(MomentsLayerResolver, self).__init__()

        # Graph sequence where keep_dims is False and dims of 1 are stripped (default)
        sequence = GraphSequence([
            ConverterSequenceNode('moments/mean', ['Mean']),
            ConverterSequenceNode('moments/StopGradient', ['StopGradient']),
            ConverterSequenceNode('moments/SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('moments/variance', ['Mean']),
            ConverterSequenceNode('moments/squeeze_mean', ['Squeeze']),
            ConverterSequenceNode('moments/squeeze_variance', ['Squeeze']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('mean_reduction_indices', ['?']),
            NonConsumableConverterSequenceNode('variance_reduction_indices', ['?']),
        ])
        sequence.set_inputs('moments/mean', ['input','mean_reduction_indices'])
        sequence.set_inputs('moments/StopGradient', ['moments/mean'])
        sequence.set_inputs('moments/SquaredDifference', ['input','moments/StopGradient'])
        sequence.set_inputs('moments/variance', ['moments/SquaredDifference','variance_reduction_indices'])
        sequence.set_inputs('moments/squeeze_mean', ['moments/mean'])
        sequence.set_inputs('moments/squeeze_variance', ['moments/variance'])
        sequence.set_outputs(['moments/squeeze_mean','moments/squeeze_variance'])

        # Graph sequence where keep_dims is True and input dimensions are maintained
        sequence_keep_dims = GraphSequence([
            ConverterSequenceNode('moments/mean', ['Mean']),
            ConverterSequenceNode('moments/StopGradient', ['StopGradient']),
            ConverterSequenceNode('moments/SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('moments/variance', ['Mean']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('variance_reduction_indices', ['?']),
            NonConsumableConverterSequenceNode('mean_reduction_indices', ['?']),
        ])
        sequence_keep_dims.set_inputs('moments/mean', ['input','mean_reduction_indices'])
        sequence_keep_dims.set_inputs('moments/StopGradient', ['moments/mean'])
        sequence_keep_dims.set_inputs('moments/SquaredDifference', ['input','moments/StopGradient'])
        sequence_keep_dims.set_inputs('moments/variance', ['moments/SquaredDifference','variance_reduction_indices'])
        sequence_keep_dims.set_outputs(['moments/mean','moments/variance'])

        self.sequences = [sequence, sequence_keep_dims]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                input_op = match['moments/mean']
                axes_op = match['mean_reduction_indices']

                axes = graph_helper.evaluate_tensor_output(axes_op.outputs[0])
                keep_dims = True
                if 'moments/squeeze_mean' in match:
                    keep_dims = False
                    mean_output_op = match['moments/squeeze_mean']
                    variance_output_op = match['moments/squeeze_variance']
                else:
                    mean_output_op = match['moments/mean']
                    variance_output_op = match['moments/variance']

                output_names = [str(mean_output_op.outputs[0].name),str(variance_output_op.outputs[0].name)]
                input_shape = graph_helper.get_op_output_shape(input_op)
                input_rank = len(input_shape)

                axes = [axes] if np.isscalar(axes) else axes.tolist()
                for i in range(len(axes)):
                    axes[i] = int(axes[i])
                    if axes[i] < 0:
                        axes[i] += input_rank

                descriptors.append(MomentsLayerResolver.Descriptor(str(mean_output_op.name), match.consumed_nodes, axes, 
                                                                   keep_dims, output_names=output_names))

        return descriptors


class MomentsLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [snpe.converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [snpe.converters.tensorflow.common.LayerDescriptor]
        :type converter_context: snpe.converters.tensorflow.converter.ConverterContext
        :type descriptor: ReductionLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        return converter_context.model.add_moments_layer(name=descriptor.layer_name,
                                                         input_name=input_name,
                                                         output_names=descriptor.output_names,
                                                         axes=descriptor.axes,
                                                         keep_dims=descriptor.keep_dims)

