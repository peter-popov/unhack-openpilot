#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.common.utils import code_to_message
from snpe.converters.tensorflow import util
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from snpe.converters.tensorflow.layers.constant import ConstantLayerResolver
from snpe.converters.tensorflow.util import ConverterError


class NonMaxSuppressionLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, max_output_size, iou_threshold, score_threshold, nms_op, boxes_op, scores_ops, input_boxes_op, input_scores_op, output_names=None):
            super(NonMaxSuppressionLayerResolver.Descriptor, self).__init__('NonMaxSuppression', name, nodes, output_names=output_names)
            self.max_output_size = max_output_size
            self.iou_threshold = iou_threshold
            self.score_threshold = score_threshold

            self.nms_op = nms_op
            self.boxes_op = boxes_op
            self.scores_ops = scores_ops

            # Input
            self.input_boxes_op = input_boxes_op
            self.input_scores_op = input_scores_op

            # Output
            self.output_boxes_op = None
            self.output_scores_op = None
            self.output_features_op = []

        def is_output_op(self, op):
            if op in self.output_features_op:
                return True
            elif op == self.output_boxes_op or op == self.output_scores_op:
                return True
            else:
                return False

    def __init__(self):

        # Seq #1 : with boxes reshaped and scores reshaped and sliced, , for layer nms/NonMaxSuppressionV2
        sequence_1 = GraphSequence([
            # Multi class scores
            NonConsumableConverterSequenceNode('scores_input', ['?']),
            ConverterSequenceNode('scores_reshape', ['Reshape']),
            ConverterSequenceNode('scores_reshape_input_shape', ['?']),
            ConverterSequenceNode('strided_slice_input_beign', ['?']),
            ConverterSequenceNode('strided_slice_input_end', ['?']),
            ConverterSequenceNode('strided_slice_input_strides', ['?']),
            ConverterSequenceNode('scores', ['StridedSlice']),

            # Boxes
            NonConsumableConverterSequenceNode('boxes_input', ['?']),
            ConverterSequenceNode('boxes', ['Reshape']),
            ConverterSequenceNode('boxes_reshape_input_shape', ['?']),

            ConverterSequenceNode('nms', ['NonMaxSuppressionV2']),
            ConverterSequenceNode('max_output_size', ['Const']),
            ConverterSequenceNode('iou_threshold', ['?']),
        ])
        sequence_1.set_inputs('boxes', ['boxes_input', 'boxes_reshape_input_shape'])
        sequence_1.set_inputs('scores_reshape', ['scores_input', 'scores_reshape_input_shape'])
        sequence_1.set_inputs('scores', ['scores_reshape', 'strided_slice_input_beign', 'strided_slice_input_end', 'strided_slice_input_strides'])
        sequence_1.set_inputs('nms', ['boxes', 'scores', 'max_output_size', 'iou_threshold'])
        sequence_1.set_outputs(['nms'])

        # Seq #2, with boxes and scores squeezed, for layer nms/NonMaxSuppressionV2
        sequence_2 = GraphSequence([
            # Multi class scores
            NonConsumableConverterSequenceNode('scores_input', ['?']),
            ConverterSequenceNode('scores', ['Squeeze']),

            # Boxes
            NonConsumableConverterSequenceNode('boxes_input', ['?']),
            ConverterSequenceNode('boxes', ['Squeeze']),

            ConverterSequenceNode('nms', ['NonMaxSuppressionV2']),
            ConverterSequenceNode('max_output_size', ['Const']),
            ConverterSequenceNode('iou_threshold', ['Const']),
        ])
        sequence_2.set_inputs('boxes', ['boxes_input'])
        sequence_2.set_inputs('scores', ['scores_input'])
        sequence_2.set_inputs('nms', ['boxes', 'scores', 'max_output_size', 'iou_threshold'])
        sequence_2.set_outputs(['nms'])

        # Seq #3,where no reshapes/slices are added (the resolver will be handling the reshapes in this case, as needed)
        sequence_3 = GraphSequence([
            NonConsumableConverterSequenceNode('boxes', ['?']),
            NonConsumableConverterSequenceNode('scores', ['?']),
            NonConsumableConverterSequenceNode('max_output_size', ['Const']),
            NonConsumableConverterSequenceNode('stub_1', ['?']),
            ConverterSequenceNode('nms', ['NonMaxSuppressionV2']),
            NonConsumableConverterSequenceNode('iou_threshold', ['?']),
        ])

        sequence_3.set_inputs('nms', ['boxes', 'scores', 'max_output_size', 'iou_threshold'])
        sequence_3.set_outputs(['nms'])

        self.sequences = [sequence_1, sequence_2, sequence_3]

        # TODO: following added for VIVO support of nms + gather in 1.23.0 to support features as inputs
        #       remove for 1.24.0 release
        # Filter seqs
        filter_sequence = GraphSequence([
            ConverterSequenceNode('gather', ['GatherV2']),
            ConverterSequenceNode('axis', ['Const']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('indices', ['NonMaxSuppressionV3'])
        ])
        filter_sequence.set_inputs('gather', ['params', 'indices', 'axis'])
        filter_sequence.set_outputs(['gather'])

        # Filter seqs 2
        filter_sequence_2 = GraphSequence([
            ConverterSequenceNode('gather', ['Gather']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('indices', ['NonMaxSuppressionV2'])
        ])
        filter_sequence_2.set_inputs('gather', ['params', 'indices'])
        filter_sequence_2.set_outputs(['gather'])

        self.g_sequences = [filter_sequence, filter_sequence_2]

    # TODO: following added for VIVO support of nms + gather in 1.23.0 to support features as inputs
    #       remove for 1.24.0 release
    def _resolve_for_gather_layer(self, graph_matcher, graph_helper, descriptor):
        for sequence in self.g_sequences:
            for match in graph_matcher.match_sequence(sequence):
                # Filter ops use nms as input.
                if match['indices'] != descriptor.nms_op:
                    continue

                params_op = match['params']
                output_op = match['gather']

                if params_op == descriptor.boxes_op or params_op == descriptor.input_boxes_op:
                    descriptor.output_boxes_op = output_op
                elif params_op in descriptor.scores_ops:
                    descriptor.output_scores_op = output_op
                else:
                    descriptor.output_features_op.append(output_op)

                descriptor.child_ops.extend(match.consumed_nodes)

        # Validation
        if not (descriptor.output_boxes_op and descriptor.output_scores_op):
            raise ConverterError('Cannot find bboxes or scores')

        # Order is important
        output_names = [str(descriptor.output_boxes_op.outputs[0].name),
                        str(descriptor.output_scores_op.outputs[0].name),
                        descriptor.layer_name + "_classes"]

        for feature_output in descriptor.output_features_op:
            output_names.append(str(feature_output.outputs[0].name))

        descriptor.output_names = output_names

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):

                # resolve layer for nms operation
                nms_op = match['nms']
                boxes_op = match['boxes']
                scores_ops = [match[k] for k in match.keys() if k.startswith("score")]

                input_boxes_op = match['boxes_input'] if 'boxes_input' in match else boxes_op
                input_scores_op = match['scores_input'] if 'scores_input' in match else match['scores']

                max_output_size = graph_helper.evaluate_tensor_output(match['max_output_size'].outputs[0])
                iou_threshold = graph_helper.evaluate_tensor_output(match['iou_threshold'].outputs[0])
                score_threshold = graph_helper.evaluate_tensor_output(match['score_threshold']) if 'score_threshold' in match else 0

                consumed_nodes = match.consumed_nodes

                nms_descriptor = NonMaxSuppressionLayerResolver.Descriptor(
                    str(nms_op.name), consumed_nodes, max_output_size, iou_threshold, score_threshold, nms_op, boxes_op,
                    scores_ops, input_boxes_op, input_scores_op, output_names=[str(nms_op.outputs[0].name)])

                descriptors.extend([nms_descriptor])

                # TODO: following added for VIVO support of nms + gather in 1.23.0 to support features as inputs
                #       remove for 1.24.0 release
                # resolve layer for gather operation
                self._resolve_for_gather_layer(graph_matcher, graph_helper, nms_descriptor)

                if input_boxes_op.type == 'Const':
                    boxes_tensor = graph_helper.evaluate_tensor_output(input_boxes_op.outputs[0])
                    boxes_shape = graph_helper.get_op_output_shape(input_boxes_op)
                    if len(boxes_shape) == 2:
                        boxes_shape.insert(0, 1)
                    else:
                        raise ConverterError(code_to_message.get_error_message('ERROR_TF_NMS_BOXES_SHAPE'), len(boxes_shape))
                    const_descriptor = ConstantLayerResolver.Descriptor(str(input_boxes_op.name), [input_boxes_op],
                                                                        boxes_tensor, boxes_shape, nms_descriptor)
                    descriptors.append(const_descriptor)

        return descriptors


class NonMaxSuppressionLayerBuilder(LayerBuilder):

    def _get_inputs_for_nms(self, op):

        input_op = op.name
        nodes_dict = op.graph._nodes_by_name

        if op.type == 'Reshape' or op.type == 'Squeeze':
            parent_input_op = op.inputs[0]
            if len(parent_input_op.shape) == 3:
                input_op = parent_input_op.name
        elif op.type == 'StridedSlice':
            parent_input_op = op.inputs[0]
            parent_input_name = parent_input_op.name.replace(":0", "")
            input_op = self._get_inputs_for_nms(nodes_dict[parent_input_name])

        return input_op

    def _build_input_layers(self, converter_context, descriptor, names):
        """
        This function helps to reshape the inputs of tf.image.non_max_suppression to align with
        what SNPE expects multiclassnms.
        """
        for op in names:
            input_shape = converter_context.graph_helper.get_op_output_shape(op)
            if len(input_shape) < 3:
                input_name = names[op]
                intermediate_output_name = input_name + '_nms_reshape_to_3d'
                names[op] = intermediate_output_name

                if len(input_shape) == 2 and input_shape[0] == 1:
                    input_shape.append(1)
                else:
                    # Add separate case to scores when it is 1 dimensional so we want to
                    # append and pre-prepend a dimension to align second dim with num of boxes.
                    if "score" in intermediate_output_name and len(input_shape) == 1:
                        input_shape.append(1)
                    input_shape = util.expand_to_rank(input_shape, 3)

                converter_context.model.add_reshape_layer(input_name + '_pre_reshape',
                                                          input_shape,
                                                          input_name,
                                                          intermediate_output_name)

    @staticmethod
    def _compare_op_shapes(converter_context, ops):
        """
        Compares the shape of all ops in the list
        :param ops: list of ops
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :return: True if all are equal or empty list, False otherwise
        """
        if len(ops):
            shape = converter_context.graph_helper.get_op_output_shape(ops[0])  # get shape for first op
            for op in ops:
                if shape != converter_context.graph_helper.get_op_output_shape(op):
                    return False
        else:
            print("WARNING: empty list provided to compare nms ops shapes")
        return True

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: NonMaxSuppressionLayerResolver.Descriptor
        :rtype: int
        """

        names = {}
        for input_descriptor in input_descriptors:
            if input_descriptor.is_output_op(descriptor.input_boxes_op):
                names[descriptor.input_boxes_op] = input_descriptor.output_names[0]
            elif input_descriptor.is_output_op(descriptor.input_scores_op):
                names[descriptor.input_scores_op] = input_descriptor.output_names[0]

        if len(names) != 2:
            raise ConverterError("Failed to detect inputs for nms op.")

        input_names = [names[descriptor.input_boxes_op], names[descriptor.input_scores_op]]
        input_names.extend(list(set(self.get_input_names(converter_context, descriptor, input_descriptors)) - set(input_names)))

        # input/output ops list
        input_output_ops_pairs = [(descriptor.input_boxes_op, descriptor.output_boxes_op),
                                  (descriptor.input_scores_op, descriptor.output_scores_op)]

        # add reshape input layers as needed to input layers to work with snpe multiclassnms layer
        self._build_input_layers(converter_context, descriptor, names)

        input_names[0] = names[descriptor.input_boxes_op]
        input_names[1] = names[descriptor.input_scores_op]
        output_names = descriptor.output_names[:]

        # adding suffix for boxes and scores since we need to do post reshape(below) to get back to TF shape
        for input_op, output_op in input_output_ops_pairs:
            for i in range(0, len(output_names)):
                if output_names[i] == output_op.outputs[0].name:
                    output_names[i] = output_names[i] + "_intermediate"

        converter_context.model.add_multi_class_nms_layer(name=descriptor.layer_name,
                                                          input_names=input_names,
                                                          output_names=output_names,
                                                          scoreThreshold=descriptor.score_threshold,
                                                          iouThreshold=descriptor.iou_threshold,
                                                          maxDetectionPerClass=descriptor.max_output_size,
                                                          maxTotalDetections=descriptor.max_output_size)

        # Post-processing, revert back reshaped layers to the expected output shape from Tensorflow
        for input_op, output_op in input_output_ops_pairs:
            for i in range(0, len(output_names)):
                if output_op.outputs[0].name in output_names[i]:
                    output_name = output_op.outputs[0].name
                    shape = converter_context.graph_helper.get_op_output_shape(output_op)
                    converter_context.model.add_reshape_layer(output_name + '_post_reshape_to_' + str(len(shape)) + 'd',
                                                              shape,
                                                              output_names[i],
                                                              output_name)
