#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.sequences.ignored import (
    ignored_sequence_1,
    ignored_sequence_2,
    dropout_cell_sequence,
    real_div_sequence,
    identity_sequence,
    placeholder_with_default_sequence,
    batchnorm_fold_sequence,
    batchnorm_fold_sequence_reshape
)


class IgnoredLayersResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes):
            super(IgnoredLayersResolver.Descriptor, self).__init__('IgnoredLayer', name, nodes)
            # define pattern one to be ignored

    def __init__(self):
        self.sequences = [
            ignored_sequence_1,
            ignored_sequence_2,
            dropout_cell_sequence,
            real_div_sequence,
            identity_sequence,
            placeholder_with_default_sequence,
            batchnorm_fold_sequence,
            batchnorm_fold_sequence_reshape
        ]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for pattern_output_nodes in self.sequences:
            matches = graph_matcher.match_sequence(pattern_output_nodes)
            if len(matches) == 0:
                continue

            for match in matches:
                consumed_nodes = match.consumed_nodes
                d = IgnoredLayersResolver.Descriptor(str(consumed_nodes[0].name), consumed_nodes)
                descriptors.append(d)

        return descriptors


class IgnoredLayersBuilder(LayerBuilder):

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        descriptor.set_ignored(True)

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        return None
