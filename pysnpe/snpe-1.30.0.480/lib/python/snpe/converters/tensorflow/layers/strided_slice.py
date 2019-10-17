#!/usr/bin/env python
# // =============================================================================
# //
# // Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
# // All Rights Reserved.
# // Confidential and Proprietary - Qualcomm Technologies, Inc.
# //
# // =============================================================================
import numpy as np

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from snpe.converters.tensorflow.util import ConverterError


class StridedSliceLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_shape, begin, end, strides, begin_mask, end_mask,
                     ellipsis_mask, new_axis_mask, shrink_axis_mask, output_names=None):
            super(StridedSliceLayerResolver.Descriptor, self).__init__('StridedSlice', name, nodes, output_names=output_names)
            self.input_shape = input_shape
            self.begin = begin
            self.end = end
            self.strides = strides
            self.begin_mask = begin_mask
            self.end_mask = end_mask
            self.ellipsis_mask = ellipsis_mask
            self.new_axis_mask = new_axis_mask
            self.shrink_axis_mask = shrink_axis_mask

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['StridedSlice']),
            ConverterSequenceNode('begin', ['Const']),
            ConverterSequenceNode('end', ['Const']),
            ConverterSequenceNode('strides', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'begin', 'end', 'strides'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []

        for match in graph_matcher.match_sequence(self.sequence):
            strided_slice_op = match['root']
            input_op = match['input']

            if input_op.type == "Const":
                continue

            begin_op = match['begin']
            end_op = match['end']
            strides_op = match['strides']

            begin_tensor = graph_helper.evaluate_tensor_output(begin_op.outputs[0])
            end_tensor = graph_helper.evaluate_tensor_output(end_op.outputs[0])
            strides_tensor = graph_helper.evaluate_tensor_output(strides_op.outputs[0])
            input_tensor = graph_helper.evaluate_tensor_output(input_op.outputs[0])

            begin_shape = graph_helper.get_op_output_shape(begin_op)
            end_shape = graph_helper.get_op_output_shape(end_op)
            strides_shape = graph_helper.get_op_output_shape(strides_op)
            input_shape = graph_helper.get_op_output_shape(input_op)

            if begin_shape != end_shape or begin_shape != strides_shape:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_STRIDED_SLICE_SHAPE_MISMATCH'))

            begin_mask = strided_slice_op.get_attr("begin_mask")
            end_mask = strided_slice_op.get_attr("end_mask")
            ellipsis_mask = strided_slice_op.get_attr("ellipsis_mask")
            new_axis_mask = strided_slice_op.get_attr("new_axis_mask")
            shrink_axis_mask = strided_slice_op.get_attr("shrink_axis_mask")

            consumed_nodes = match.consumed_nodes
            pad_descriptor = StridedSliceLayerResolver.Descriptor(
                str(strided_slice_op.name), consumed_nodes, input_shape,
                begin_tensor, end_tensor, strides_tensor, begin_mask, end_mask, ellipsis_mask,
                new_axis_mask, shrink_axis_mask, output_names=[str(strided_slice_op.outputs[0].name)])
            descriptors.extend([pad_descriptor])

        return descriptors


class StridedSliceLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: StridedSliceLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]

        if descriptor.ellipsis_mask != 0 or descriptor.new_axis_mask != 0:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_STRIDED_SLICE_UNSUPPORTED_MASKS'))

        input_rank = len(descriptor.input_shape)
        strides_rank = descriptor.strides.shape[0]

        # Extend to match input rank
        begin = np.append(descriptor.begin, np.zeros(input_rank - strides_rank, dtype=np.int32)).tolist()
        strides = np.append(descriptor.strides, np.ones(input_rank - strides_rank, dtype=np.int32)).tolist()
        end = np.append(descriptor.end, descriptor.input_shape[strides_rank:]).astype(np.int32).tolist()

        # Apply the binary masks
        for i in range(len(strides)):
            begin_mask_bit = self.get_bit(descriptor.begin_mask, i)
            end_mask_bit = self.get_bit(descriptor.end_mask, i)
            shrink_mask_bit = self.get_bit(descriptor.shrink_axis_mask, i)

            # Convert negative indices
            if begin[i] < 0:
                begin[i] += descriptor.input_shape[i]
            if end[i] < 0:
                end[i] += descriptor.input_shape[i]

            # Apply mask bits
            if strides[i] > 0:
                if begin_mask_bit:
                    begin[i] = 0
                if end_mask_bit:
                    end[i] = descriptor.input_shape[i]
            else:
                if begin_mask_bit:
                    begin[i] = descriptor.input_shape[i] - 1
                if end_mask_bit:
                    end[i] = -1

            # Apply shrink_axis_mask
            if shrink_mask_bit:
                strides[i] = 1
                end[i] = begin[i] + strides[i]

        return converter_context.model.add_strided_slice_layer(name=descriptor.layer_name,
                                                               input_name=input_name,
                                                               output_name=output_name,
                                                               begin=begin,
                                                               end=end,
                                                               strides=strides,
                                                               shrink_axis_mask=descriptor.shrink_axis_mask)

    @classmethod
    def get_bit(cls, val, i):
        return val & (1 << i)
