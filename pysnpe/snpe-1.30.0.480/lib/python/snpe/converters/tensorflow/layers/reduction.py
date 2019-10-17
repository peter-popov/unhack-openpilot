#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from abc import ABCMeta
from abc import abstractmethod
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class ReductionLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, axes, keep_dims, output_names=None):
            super(ReductionLayerResolver.Descriptor, self).__init__(layer_type, name, nodes, output_names=output_names)
            self.axes = axes
            self.keep_dims = keep_dims

    def __init__(self, layer_type, op_type, descriptor_class):
        super(ReductionLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            ConverterSequenceNode('reduction_indices', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'reduction_indices'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            reduction_op = match['root']
            input_op = match['input']
            reduction_indices_op = match['reduction_indices']

            axes = graph_helper.evaluate_tensor_output(reduction_indices_op.outputs[0])
            keep_dims = bool(reduction_op.get_attr('keep_dims'))

            input_shape = graph_helper.get_op_output_shape(input_op)
            input_rank = len(input_shape)

            axes = [axes] if np.isscalar(axes) else axes.tolist()
            for i in range(len(axes)):
                axes[i] = int(axes[i])
                if axes[i] < 0:
                    axes[i] += input_rank

            reduction_descriptor = self._descriptor_class(self._layer_type, str(reduction_op.name),
                                                          match.consumed_nodes, axes, keep_dims,
                                                          output_names=[str(reduction_op.outputs[0].name)])
            descriptors.extend([reduction_descriptor])

        return descriptors


class ReductionLayerBuilder(LayerBuilder):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        pass


class ReductionMeanLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionMeanLayerResolver, self).__init__('ReduceMean', 'Mean', ReductionMeanLayerResolver.Descriptor)


class ReductionMeanLayerBuilder(ReductionLayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReductionLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_reduction_mean_layer(name=descriptor.layer_name,
                                                                input_name=input_name,
                                                                output_name=output_name,
                                                                axes=descriptor.axes,
                                                                keep_dims=descriptor.keep_dims)


class ReductionProdLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionProdLayerResolver, self).__init__('ReduceProd', 'Prod', ReductionProdLayerResolver.Descriptor)


class ReductionProdLayerBuilder(ReductionLayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReductionLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_reduction_prod_layer(name=descriptor.layer_name,
                                                                input_name=input_name,
                                                                output_name=output_name,
                                                                axes=descriptor.axes,
                                                                keep_dims=descriptor.keep_dims)


class ReductionSumLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionSumLayerResolver, self).__init__('ReduceSum', 'Sum', ReductionSumLayerResolver.Descriptor)


class ReductionSumLayerBuilder(ReductionLayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReductionLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_reduction_sum_layer(name=descriptor.layer_name,
                                                               input_name=input_name,
                                                               output_name=output_name,
                                                               axes=descriptor.axes,
                                                               keep_dims=descriptor.keep_dims)


class ReductionMinLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionMinLayerResolver, self).__init__('ReduceMin', 'Min', ReductionMinLayerResolver.Descriptor)


class ReductionMinLayerBuilder(ReductionLayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReductionLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_reduction_min_layer(name=descriptor.layer_name,
                                                                input_name=input_name,
                                                                output_name=output_name,
                                                                axes=descriptor.axes,
                                                                keep_dims=descriptor.keep_dims)


class ReductionMaxLayerResolver(ReductionLayerResolver):
    class Descriptor(ReductionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(ReductionMaxLayerResolver, self).__init__('ReduceMax', 'Max', ReductionMaxLayerResolver.Descriptor)


class ReductionMaxLayerBuilder(ReductionLayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReductionLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_reduction_max_layer(name=descriptor.layer_name,
                                                                input_name=input_name,
                                                                output_name=output_name,
                                                                axes=descriptor.axes,
                                                                keep_dims=descriptor.keep_dims)
