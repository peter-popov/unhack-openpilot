#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class FakeQuantLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, is_act_quant, min, max):
            super(FakeQuantLayerResolver.Descriptor, self).__init__(layer_type, name, nodes)

            self.is_act_quant = is_act_quant
            self.min = min
            self.max = max

        @property
        def output_names(self):
            return [str(self.child_ops[0].outputs[0].name)]

        def is_output_op(self, op):
            return op in self.child_ops

        def get_output_names_for(self, input_tensors):
            return self.output_names

    def __init__(self):
        self.sequence = GraphSequence([
                            ConverterSequenceNode('root', ['FakeQuantWithMinMaxVars']),
                            ConverterSequenceNode('min', ['?']),
                            ConverterSequenceNode('max', ['?']),
                            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        self.sequence.set_inputs('root', ['input', 'min', 'max'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)

        # Nothing matched
        if len(matches) == 0:
            return []

        potential_descriptors = []
        for match in matches:
            fake_quant_op = match['root']
            min_op = match['min']
            max_op = match['max']
            input_op = match['input']

            # It's not activation-fake-quant node if input type is const
            is_act_quant = False if input_op.type == 'Identity' else True
            min = self._get_float(graph_helper, min_op)
            max = self._get_float(graph_helper, max_op)

            consumed_nodes = match.consumed_nodes

            potential_descriptors.append(
                FakeQuantLayerResolver.Descriptor('FakeQuant', str(fake_quant_op.name), consumed_nodes, is_act_quant, min, max))

        return potential_descriptors

    def _get_float(self, graph_helper, op):
        tensor = graph_helper.get_tensor_by_name(op.name)
        return graph_helper.evaluate_tensor_output(tensor)


class FakeQuantLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FakeQuantLayerResolver.Descriptor
        :rtype: int
        """
        return None

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):

        descriptor.set_ignored(True)

        # Only fuse activation-quant layer
        if not descriptor.is_act_quant:
            return

        converter_context.replace_layer_input_with(output_descriptors[0], descriptor, input_descriptors)



