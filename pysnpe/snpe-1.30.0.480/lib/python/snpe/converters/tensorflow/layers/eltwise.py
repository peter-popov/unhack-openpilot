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
from snpe.converters.tensorflow.util import ConverterError
from snpe.converters.tensorflow.layers.constant import ConstantLayerResolver
from abc import ABCMeta
from abc import abstractmethod
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class EltWiseLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    def __init__(self, layer_type, op_type, descriptor_class):
        super(EltWiseLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            NonConsumableConverterSequenceNode('input1', ['?']),
            NonConsumableConverterSequenceNode('input2', ['?'])
        ])
        self.sequence.set_inputs('root', ['input1', 'input2'])
        self.sequence.set_outputs(['root'])

        self.sequence_with_identity = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            ConverterSequenceNode('identity', ['Identity']),
            NonConsumableConverterSequenceNode('input1', ['?']),
            NonConsumableConverterSequenceNode('input2', ['?'])
        ])
        self.sequence_with_identity.set_inputs('identity', ['root'])
        self.sequence_with_identity.set_inputs('root', ['input1', 'input2'])
        self.sequence_with_identity.set_outputs(['identity'])

        self.sequence_with_const_input = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            NonConsumableConverterSequenceNode('const', ['Const', 'Identity']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.sequence_with_const_input.set_inputs('root', ['const', 'other'])
        self.sequence_with_const_input.set_outputs(['root'])

        self.sequence_with_const_input_and_identity = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            ConverterSequenceNode('identity', ['Identity']),
            NonConsumableConverterSequenceNode('const', ['Const', 'Identity']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.sequence_with_const_input_and_identity.set_inputs('root', ['const', 'other'])
        self.sequence_with_const_input_and_identity.set_inputs('identity', ['root'])
        self.sequence_with_const_input_and_identity.set_outputs(['identity'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        non_const_input_sequences = [self.sequence_with_identity, self.sequence]
        for sequence in non_const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes)
                descriptors.append(descriptor)

        const_input_sequences = [self.sequence_with_const_input_and_identity, self.sequence_with_const_input]
        for sequence in const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                eltwise_descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name),
                                                            match.consumed_nodes)
                descriptors.append(eltwise_descriptor)

                const_op = match['const']
                const_consumed_ops = [const_op]
                while const_op.type == 'Identity':
                    const_op = const_op.inputs[0].op
                    const_consumed_ops.append(const_op)

                if const_op.type != 'Const':
                    continue

                const_tensor = graph_helper.evaluate_tensor_output(const_op.outputs[0])
                eltwise_shape = graph_helper.get_op_output_shape(eltwise_op)

                # Do not broadcast the constant for Sub, RealDiv, Mul ops because they support
                # dynamic broadcasting during runtime. This if statement should be removed once
                # all runtimes support dynamic broadcasting.
                if self._op_type in ['Sub', 'RealDiv', 'Mul']:
                    eltwise_shape = graph_helper.get_op_output_shape(const_op)
                    if not eltwise_shape:
                        eltwise_shape = [1]
                else:
                    if len(eltwise_shape) > 4:
                        eltwise_shape = eltwise_shape[-4:]
                    if len(eltwise_shape) > 3:
                        broadcast_shape = eltwise_shape[1:]
                    else:
                        broadcast_shape = eltwise_shape

                    if list(const_tensor.shape) != broadcast_shape:
                        const_tensor = self._broadcast_tensor(const_tensor, broadcast_shape)

                const_descriptor = ConstantLayerResolver.Descriptor(str(const_op.name), const_consumed_ops,
                                                                    const_tensor, eltwise_shape, eltwise_descriptor)
                descriptors.append(const_descriptor)

        return descriptors

    def _broadcast_tensor(self, tensor, shape):
        raise ConverterError('ElementWise resolver must implement broadcast method.')


class EltWiseLayerBuilder(LayerBuilder):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        pass


class EltWiseSumLayerResolver(EltWiseLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseSumLayerResolver, self).__init__('ElementWiseSum', 'Add', EltWiseSumLayerResolver.Descriptor)

    def _broadcast_tensor(self, tensor, shape):
        broadcasted_tensor = np.zeros(shape, dtype=np.float32)
        broadcasted_tensor = broadcasted_tensor + tensor
        return broadcasted_tensor


class EltWiseSumLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseSumLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_sum_layer(descriptor.layer_name,
                                                                 [1.0 for _ in input_names],
                                                                 input_names,
                                                                 output_name)


class EltWiseSubLayerResolver(EltWiseLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseSubLayerResolver, self).__init__('ElementWiseSub', 'Sub', EltWiseSubLayerResolver.Descriptor)


class EltWiseSubLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseSubLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_binary_sub_layer(descriptor.layer_name,
                                                                        input_names,
                                                                        output_name)


class EltWiseMulLayerResolver(EltWiseLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseMulLayerResolver, self).__init__('ElementWiseMul', 'Mul', EltWiseMulLayerResolver.Descriptor)


class EltWiseMulLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseMulLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_binary_product_layer(descriptor.layer_name,
                                                                     input_names,
                                                                     output_name)


class EltWiseMaxLayerResolver(EltWiseLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseMaxLayerResolver, self).__init__('ElementWiseMax', 'Maximum', EltWiseMaxLayerResolver.Descriptor)

    def _broadcast_tensor(self, tensor, shape):
        broadcasted_tensor = np.zeros(shape, dtype=np.float32)
        broadcasted_tensor = broadcasted_tensor + tensor
        return broadcasted_tensor


class EltWiseMaxLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseMaxLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_max_layer(descriptor.layer_name,
                                                                 input_names,
                                                                 output_name)


class EltWiseDivLayerResolver(EltWiseLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseDivLayerResolver, self).__init__('ElementWiseDiv', 'RealDiv', EltWiseDivLayerResolver.Descriptor)

    def _broadcast_tensor(self, tensor, shape):
        broadcasted_tensor = np.zeros(shape, dtype=np.float32)
        broadcasted_tensor = broadcasted_tensor + tensor
        return broadcasted_tensor


class EltWiseDivLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseDivLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_binary_div_layer(descriptor.layer_name,
                                                                        input_names,
                                                                        output_name)

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):

        constant_input_descriptor = [d for d in input_descriptors if isinstance(d, ConstantLayerResolver.Descriptor)]
        if len(constant_input_descriptor) == 1 and np.all(constant_input_descriptor[0].value == 1):
            descriptor.set_ignored(True)
            constant_input_descriptor[0].set_ignored(True)
