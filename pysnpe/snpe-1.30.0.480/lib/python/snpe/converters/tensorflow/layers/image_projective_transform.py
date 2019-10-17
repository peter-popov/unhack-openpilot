#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import tensorflow.contrib

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
)
from snpe.converters.tensorflow.util import ConverterError


class ImageProjectiveTransformLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, interpolation_mode, output_names=None):
            super(ImageProjectiveTransformLayerResolver.Descriptor, self).__init__(
                'ImageProjectiveTransform', name, operations, output_names=output_names)
            self.interpolation_mode = interpolation_mode

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['ImageProjectiveTransform'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        matches = graph_matcher.match_sequence(self.sequence)
        for match in matches:
            image_proj_transform = match['root']

            output_op_nodes_names = [str(image_proj_transform.outputs[0].name)]
            consumed_nodes = match.consumed_nodes

            interpolation = str(image_proj_transform.get_attr('interpolation').decode('utf-8'))
            if interpolation == "BILINEAR":
               interpolation_mode = 0
            elif interpolation == "NEAREST":
                interpolation_mode = 1
            else:
                raise ConverterError(
                    code_to_message.get_error_message('ERROR_TF_RESOLVE_IMAGE_TRANSFORM_INTERPOLATION'))

            potential_descriptors.append(
                ImageProjectiveTransformLayerResolver.Descriptor(str(image_proj_transform.name),
                                                                 consumed_nodes, interpolation_mode,
                                                                 output_names=output_op_nodes_names)
            )
        return potential_descriptors


class ImageProjectiveTransformLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ImageProjectiveTransformLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_image_projective_transform_layer(name=descriptor.layer_name,
                                                                            input_names=input_names,
                                                                            output_name=output_name,
                                                                            interpolation=descriptor.interpolation_mode)
