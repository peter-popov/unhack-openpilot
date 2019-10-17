#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from snpe.converters.tensorflow.layers.constant import ConstantLayerResolver
from snpe.converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from snpe.converters.tensorflow.layers.lstm import LstmLayerResolver
from snpe.converters.tensorflow.util import ConverterError
from snpe.converters.tensorflow.util import GraphHelper


class ConcatLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, output_names=None):
            super(ConcatLayerResolver.Descriptor, self).__init__('Concatenation', name, nodes,
                                                                 output_names=output_names)
            self.axis = axis

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Concat', 'ConcatV2'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            concat_op = match['root']
            consumed_nodes = match.consumed_nodes
            concat_descriptor = ConcatLayerResolver.Descriptor(str(concat_op.name), consumed_nodes,
                                                               None, [concat_op.outputs[0].name])

            non_const_inputs = [tensor for tensor in concat_op.inputs if tensor.op.type != 'Const']
            const_ops = [tensor.op for tensor in concat_op.inputs if tensor.op.type == 'Const']
            axis_tensor = None
            if len(non_const_inputs) < 2 or len(const_ops) > 1:
                for i in range(0, len(const_ops) - 1):
                    const_value = graph_helper.evaluate_tensor_output(const_ops[i].outputs[0])
                    const_shape = graph_helper.get_op_output_shape(const_ops[i].outputs[0])
                    descriptors.append(ConstantLayerResolver.Descriptor(str(const_ops[i]),
                                                                        [const_ops[i]],
                                                                        const_value,
                                                                        const_shape,
                                                                        concat_descriptor))
                # Make the assumption that the axis is always the last constant
                axis_tensor = const_ops[-1]

            max_shape = 0
            for t in non_const_inputs:
                shape = graph_helper.get_op_output_shape(t.op)
                if len(shape) > max_shape:
                    max_shape = len(shape)

            if not axis_tensor:
                axis_tensor = GraphHelper.filter_single_op_by_type([t.op for t in concat_op.inputs], 'Const')
            axis = int(graph_helper.evaluate_tensor_output(axis_tensor.outputs[0]))
            if axis < 0:
                axis += max_shape

            concat_descriptor.axis = axis
            descriptors.append(concat_descriptor)

        return descriptors


class ConcatLayerBuilder(LayerBuilder):

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        # Optimization to avoid going to 5-Dimensional Concat only if batch input is 1.
        # Check the following to see if the optimization can be made
        # 1. Input op must be ExpandDims
        # 2. Axis of ExpandDims must match Axis of Concat
        # 3. Input data tensor to ExpandDims must have batch = 1
        # 4. Output of Concat must go to a reshape
        # 5. Reshape must be merging the batch and 5-th Dimension together
        get_tensor = converter_context.graph_helper.get_op_input_tensors
        evaluate_output = converter_context.graph_helper.evaluate_tensor_output
        get_shape = converter_context.graph_helper.get_op_output_shape
        if all(x.child_ops[-1].type == 'ExpandDims' and
               evaluate_output(get_tensor(x.child_ops[-1], ('?', 'Const'))[1]) == descriptor.axis and # Check ExpandDims axis == Concat Axis
               get_shape(get_tensor(x.child_ops[-1], ('?', 'Const'))[0])[0] == 1 # Check input batch == 1
               for x in input_descriptors) and \
           len(output_descriptors) == 1 and output_descriptors[0].child_ops[-1].type == 'Reshape' and \
           len(get_shape(output_descriptors[0].child_ops[-1])) == 4:
            for input_descriptor in input_descriptors:
                input_descriptor.set_ignored(True)

            output_descriptors[0].set_ignored(True)
            descriptor.axis = 0
            return

        if len(input_descriptors) == 1 and isinstance(input_descriptors[0], IgnoredLayersResolver.Descriptor):
            descriptor.set_ignored(True)
            return

        lstm_inputs = [d for d in input_descriptors if
                       isinstance(d, LstmLayerResolver.UnrolledTimeStepDescriptor) or
                       isinstance(d, LstmLayerResolver.StateDescriptor)]
        if lstm_inputs == input_descriptors:
            converter_context.merge_descriptors(descriptor, lstm_inputs[0])
            return

        concat_outputs = [d for d in output_descriptors if isinstance(d, ConcatLayerResolver.Descriptor)]
        if concat_outputs == output_descriptors and len(concat_outputs) == 1:
            concat_on_concat_output = concat_outputs[0]
            if descriptor.axis == concat_on_concat_output.axis:
                converter_context.merge_descriptors(descriptor, concat_on_concat_output)


    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        if len(input_descriptors) < 2:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONCAT_INPUT'))

        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        return converter_context.model.add_concatenation_layer(descriptor.layer_name,
                                                               input_names,
                                                               descriptor.output_names[0],
                                                               descriptor.axis)
