#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.tensorflow.layers.constant import ConstantLayerResolver
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.util import GraphHelper
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class CropAndResizeLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, crop_height, crop_width, interpolation_method,
                     extrapolation_value, output_names=None):
            super(CropAndResizeLayerResolver.Descriptor, self).__init__('CropAndResize', name,
                                                               nodes, output_names=output_names)
            self.crop_height = crop_height
            self.crop_width = crop_width
            self.interpolation_method = interpolation_method
            self.extrapolation_value = extrapolation_value

    def __init__(self):
        sequence_crop_and_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('boxes', ['?']),
            NonConsumableConverterSequenceNode('box_ind', ['?']),
            NonConsumableConverterSequenceNode('crop_size', ['?']),
            ConverterSequenceNode('crop_and_resize', ['CropAndResize']),
        ])
        sequence_crop_and_resize.set_inputs('crop_and_resize', ['input', 'boxes', 'box_ind', 'crop_size'])
        sequence_crop_and_resize.set_outputs(['crop_and_resize'])

        self.sequences = [sequence_crop_and_resize]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                crop_and_resize = match['crop_and_resize']

                try:
                   _, _, box_ind, crop_size = GraphHelper.get_op_input_tensors(crop_and_resize, ('?', '?', '?', 'Const'))
                except TensorNotFoundError:
                    raise ConverterError(
                        code_to_message.get_message('ERROR_TF_RESOLVE_CROP_AND_RESIZE_SIZE_NOT_CONST'))

                crop_size_value = graph_helper.evaluate_tensor_output(crop_size)
                if crop_size_value.size != 2:
                    raise ConverterError(
                        code_to_message.get_message('ERROR_TF_RESOLVE_CROP_AND_RESIZE_SIZE'))

                consumed_nodes = match.consumed_nodes

                interpolation_method = str(crop_and_resize.get_attr('method'))
                extrapolation_value = float(crop_and_resize.get_attr('extrapolation_value'))

                crop_and_resize_descriptor = CropAndResizeLayerResolver.Descriptor(
                    str(crop_and_resize.name), consumed_nodes, crop_size_value[1],
                    crop_size_value[0], interpolation_method, extrapolation_value)
                potential_descriptors.append(crop_and_resize_descriptor)

                if box_ind.op.type == 'Const':
                    box_ind_value = graph_helper.evaluate_tensor_output(box_ind)
                    box_ind_shape = graph_helper.get_op_output_shape(box_ind.op)
                    constant_descriptor = ConstantLayerResolver.Descriptor(str(box_ind.op),
                                                                           [box_ind.op],
                                                                           box_ind_value,
                                                                           box_ind_shape,
                                                                           crop_and_resize_descriptor)
                    potential_descriptors.append(constant_descriptor)

        return potential_descriptors


class CropAndResizeLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        return converter_context.model.add_crop_and_resize_layer(descriptor.layer_name,
                                                                 input_names=input_names,
                                                                 output_name=descriptor.output_names[0],
                                                                 crop_height=descriptor.crop_height,
                                                                 crop_width=descriptor.crop_width,
                                                                 interpolation_method=descriptor.interpolation_method,
                                                                 extrapolation_value=descriptor.extrapolation_value)
