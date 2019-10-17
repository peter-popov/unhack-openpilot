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
from snpe.converters.tensorflow.layers.constant import ConstantLayerResolver


class TileLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, multiples, output_names=None):
            super(TileLayerResolver.Descriptor, self).__init__('Tile', name, nodes, output_names=output_names)
            self.multiples = multiples

    def __init__(self):
        self.sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('tile', ['Tile']),
            NonConsumableConverterSequenceNode('multiples', ['?'])
        ])
        self.sequence.set_inputs('tile', ['input', 'multiples'])
        self.sequence.set_outputs(['tile'])

        # sequence input->shape->stridedslice->pack->tile
        self.sequence_pack = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('shape', ['Shape']),
            NonConsumableConverterSequenceNode('stridedslice', ['StridedSlice']),
            NonConsumableConverterSequenceNode('const_3', ['Const']),
            NonConsumableConverterSequenceNode('const_4', ['Const']),
            NonConsumableConverterSequenceNode('const_5', ['Const']),
            ConverterSequenceNode('tile', ['Tile']),
            NonConsumableConverterSequenceNode('tile_multiples_pack', ['Pack']),
            NonConsumableConverterSequenceNode('const_1', ['Const']),
            NonConsumableConverterSequenceNode('const_2', ['Const']),
            NonConsumableConverterSequenceNode('tile_input', ['?'])
        ])
        self.sequence_pack.set_inputs('shape', ['input'])
        self.sequence_pack.set_inputs('stridedslice', ['shape', 'const_3', 'const_4', 'const_5'])
        self.sequence_pack.set_inputs('tile_multiples_pack', ['stridedslice', 'const_1', 'const_2'])
        self.sequence_pack.set_inputs('tile', ['tile_input', 'tile_multiples_pack'])
        self.sequence_pack.set_outputs(['tile'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        non_const_sequence = [self.sequence]
        for sequence in non_const_sequence:
            for match in graph_matcher.match_sequence(sequence):
                tile_op = match['tile']
                multiples_op = match['multiples']
                values = graph_helper.evaluate_tensor_output(multiples_op.outputs[0])

                consumed_nodes = match.consumed_nodes
                tile_descriptor = TileLayerResolver.Descriptor(
                    str(tile_op.name), consumed_nodes, values,
                    output_names=[str(tile_op.outputs[0].name)])

                descriptors.extend([tile_descriptor])

        const_sequence = [self.sequence_pack]
        for sequence in const_sequence:
            for match in graph_matcher.match_sequence(sequence):
                if 'tile_multiples_pack' in match:
                    tile_op = match['tile']
                    tile_input_op = match['tile_input']
                    tile_multiples_pack_op = match['tile_multiples_pack']

                    tile_input_tensor = graph_helper.evaluate_tensor_output(tile_input_op.outputs[0])
                    tile_multiples_pack_tensor = graph_helper.evaluate_tensor_output(tile_multiples_pack_op.outputs[0])

                    consumed_nodes = match.consumed_nodes
                    tile_descriptor = TileLayerResolver.Descriptor(
                        str(tile_op.name), consumed_nodes, tile_input_tensor,
                        output_names=[str(tile_op.outputs[0].name)])

                    tile_input_consumed_ops = [tile_input_op]
                    tile_input_shape = graph_helper.get_op_output_shape(tile_input_op)

                    const_descriptor = ConstantLayerResolver.Descriptor(str(tile_input_op.name),
                                                                        tile_input_consumed_ops,
                                                                        tile_input_tensor,
                                                                        tile_input_shape, tile_descriptor)
                    descriptors.append(const_descriptor)

        return descriptors


class TileLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: TileLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_tile_layer(name=descriptor.layer_name,
                                                      multiples=(descriptor.multiples).tolist(),
                                                      input_name=input_name,
                                                      output_name=output_name)

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        # Optimization for the sequence shape->stridedslice->pack->tile
        # Since pack is the multiples in the tile OP (not the input), if the output of pack is all ones,
        # the tile OP would be passthrough. In this case, tile OP would be ignored (i.e. its input would be
        # the input of its next OP.)
        multiples_input_descriptor = descriptor.multiples
        if all(multiples_input_descriptor == 1):
            descriptor.set_ignored(True)
            input_descriptors[0].consumer = output_descriptors[0]
