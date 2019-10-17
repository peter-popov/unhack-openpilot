#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2016-2017 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.common.utils.code_to_message import get_error_message
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder, ConverterError
from snpe.converters.tensorflow.layers.constant import ConstantLayerResolver
from snpe.converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from snpe.converters.tensorflow.layers.reshape import ReshapeLayerResolver
from snpe.converters.tensorflow.sequences import ssd


class SSDDecoderResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, scale_y, scale_x, scale_h, scale_w, input_ops):
            super(SSDDecoderResolver.Descriptor, self).__init__('SSDDecoderLayer', name, nodes)
            self.scale_y = scale_y
            self.scale_x = scale_x
            self.scale_h = scale_h
            self.scale_w = scale_w
            self.input_ops = input_ops

        def is_input_op(self, op):
            return op in self.input_ops

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        sequences = [
            ssd.box_decoder_sequence,
        ]
        for sequence in sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                output_op = match['Postprocessor/Decode/transpose_1']
                anchors_op = match['Postprocessor/Tile']
                ssd_input_ops = [match['Postprocessor/Reshape_1'], match['Postprocessor/Reshape']]
                consumed_nodes = match.consumed_nodes

                scale_y = self._resolve_scale_tensor(match['Postprocessor/Decode/div'], graph_helper)
                scale_x = self._resolve_scale_tensor(match['Postprocessor/Decode/div_1'], graph_helper)
                scale_h = self._resolve_scale_tensor(match['Postprocessor/Decode/div_2'], graph_helper)
                scale_w = self._resolve_scale_tensor(match['Postprocessor/Decode/div_3'], graph_helper)

                d = SSDDecoderResolver.Descriptor(str(output_op.name), consumed_nodes,
                                                  scale_y, scale_x, scale_h, scale_w, ssd_input_ops)
                descriptors.append(d)

                anchors = graph_helper.evaluate_tensor_output(anchors_op.outputs[0])
                anchors_shape = anchors.shape
                if len(anchors_shape) > 4:
                    anchors_shape = anchors_shape[-4:]

                const_descriptor = ConstantLayerResolver.Descriptor(str(anchors_op.name), [anchors_op], anchors,
                                                                    anchors_shape, d)
                descriptors.append(const_descriptor)

        return descriptors

    @classmethod
    def _resolve_scale_tensor(cls, scale_op, graph_helper):
        _, scale_factor_tensor = graph_helper.get_op_input_tensors(scale_op, ('?', 'Const'))
        scale_factor = float(graph_helper.evaluate_tensor_output(scale_factor_tensor))
        return scale_factor


class SSDAnchorGeneratorResolver(LayerResolver, object):

    def __init__(self):
        self.scope_name_patterns = [
            'MultipleGridAnchorGenerator'
        ]

    def is_final_resolution(self):
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for pattern in self.scope_name_patterns:
            ops = [n.original_node for n in graph_matcher.graph if n.identifier.startswith(pattern)]
            if len(ops) > 0:
                d = IgnoredLayersResolver.Descriptor(str(ops[0].name), ops)
                descriptors.append(d)

        return descriptors


class SSDNmsResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, boxes_output_op, scores_output_op, classes_output_op,
                     score_threshold, iou_threshold):
            super(SSDNmsResolver.Descriptor, self).__init__('MultiClassNonMaxSuppression', name, nodes)
            self.output_ops = [boxes_output_op, scores_output_op, classes_output_op]
            self.boxes_output_op = boxes_output_op
            self.scores_output_op = scores_output_op
            self.classes_output_op = classes_output_op
            self.score_threshold = score_threshold
            self.iou_threshold = iou_threshold

        @property
        def output_names(self):
            """
            :rtype: [str]
            """
            return ['{}_boxes'.format(self.layer_name),
                    '{}_scores'.format(self.layer_name),
                    '{}_classes'.format(self.layer_name)]

        def is_output_op(self, op):
            return op in self.output_ops

        def get_output_names_for(self, input_tensors):
            """
            :type input_tensors: [tensorflow.Tensor]
            :rtype: [str]
            """
            output_names = []
            for input_tensor in input_tensors:
                if self.boxes_output_op == input_tensor.op:
                    output_names.append(self.output_names[0])
                elif self.scores_output_op == input_tensor.op:
                    output_names.append(self.output_names[1])
                elif self.classes_output_op == input_tensor.op:
                    output_names.append(self.output_names[2])
            return output_names

    def is_final_resolution(self):
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        matches = graph_matcher.match_sequence(ssd.nms_sequence)
        for match in matches:
            anchor_op = match['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_6']
            nms_scope = '/'.join(anchor_op.name.split('/')[:-2])
            nms_ops_map = {n.identifier: n.original_node
                           for n in graph_matcher.graph if n.identifier.startswith(nms_scope)}
            nms_ops = set(nms_ops_map.values())
            nms_ops.update(match.consumed_nodes)

            classes_output_op = match[
                'Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/TensorArrayGatherV3']

            output_op_names = [
                'Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3',
                'Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArrayGatherV3'
            ]

            boxes_output_op, scores_output_op = None, None
            for name in output_op_names:
                output_op = match[name]
                shape = graph_helper.get_op_output_shape(output_op)
                if shape[-1] == 4:
                    boxes_output_op = output_op
                else:
                    scores_output_op = output_op

            score_threshold = self._resolve_score_threshold(nms_ops_map, nms_scope, graph_helper)
            iou_threshold = self._resolve_iou_threshold(nms_ops_map, nms_scope, graph_helper)
            descriptors.append(SSDNmsResolver.Descriptor(nms_scope, list(nms_ops), boxes_output_op,
                                                         scores_output_op, classes_output_op,
                                                         score_threshold, iou_threshold))
        return descriptors

    @classmethod
    def _resolve_score_threshold(cls, nms_ops_map, nms_scope, graph_helper):
        nms_op_name = '{}/map/while/MultiClassNonMaxSuppression/non_max_suppression/NonMaxSuppressionV3' \
            .format(nms_scope)
        nms_op = nms_ops_map.get(nms_op_name, None)
        if nms_op is not None:
            _, _, _, iou_tensor, score_tensor = graph_helper.get_op_input_tensors(nms_op, ('?', '?', '?', 'Const', 'Const'))
            score_threshold = float(graph_helper.evaluate_tensor_output(score_tensor))
            return score_threshold

        score_threshold_op_name = '{}/map/while/MultiClassNonMaxSuppression/FilterGreaterThan/Greater/y' \
            .format(nms_scope)
        score_threshold_op = nms_ops_map.get(score_threshold_op_name, None)
        if score_threshold_op is None:
            raise ConverterError(get_error_message('ERROR_TF_SSD_NMS_CAN_NOT_RESOLVE_SCORE_THRESHOLD'))
        score_threshold = float(graph_helper.evaluate_tensor_output(score_threshold_op.outputs[0]))
        return score_threshold

    @classmethod
    def _resolve_iou_threshold(cls, nms_ops_map, nms_scope, graph_helper):
        nms_op_name = '{}/map/while/MultiClassNonMaxSuppression/non_max_suppression/NonMaxSuppressionV3' \
            .format(nms_scope)
        nms_op = nms_ops_map.get(nms_op_name, None)
        if nms_op is not None:
            _, _, _, iou_tensor, score_tensor = graph_helper.get_op_input_tensors(nms_op, ('?', '?', '?', 'Const', 'Const'))
            iou_threshold = float(graph_helper.evaluate_tensor_output(iou_tensor))
            return iou_threshold

        nms_op_name = '{}/map/while/MultiClassNonMaxSuppression/non_max_suppression/NonMaxSuppressionV2' \
            .format(nms_scope)
        nms_op = nms_ops_map.get(nms_op_name, None)
        if nms_op is None:
            raise ConverterError(get_error_message('ERROR_TF_SSD_NMS_CAN_NOT_RESOLVE_IOU'))
        _, _, _, iou_tensor = graph_helper.get_op_input_tensors(nms_op, ('?', '?', '?', 'Const'))
        iou_threshold = float(graph_helper.evaluate_tensor_output(iou_tensor))
        return iou_threshold


class SSDDecoderLayersBuilder(LayerBuilder):

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        output_name = descriptor.output_names[0]
        anchor_input = [d for d in input_descriptors if isinstance(d, ConstantLayerResolver.Descriptor)]
        boxes_input = [d for d in input_descriptors if not isinstance(d, ConstantLayerResolver.Descriptor)]
        if len(anchor_input) != 1:
            raise ConverterError(get_error_message('ERROR_TF_SSD_ANCHOR_INPUT_MISSING'))
        anchors_layer_name = anchor_input[0].output_names[0]
        boxes_layer_name = boxes_input[0].output_names[0]
        return converter_context.model.add_box_decoder_layer(output_name,
                                                             [boxes_layer_name, anchors_layer_name],
                                                             [output_name],
                                                             scale_y=descriptor.scale_y,
                                                             scale_x=descriptor.scale_y,
                                                             scale_h=descriptor.scale_h,
                                                             scale_w=descriptor.scale_w)


class SSDNmsLayersBuilder(LayerBuilder):

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        reshapes = [d for d in input_descriptors if isinstance(d, ReshapeLayerResolver.Descriptor)]
        if len(reshapes) == 1 and self.is_expanding_boxes_input(converter_context, reshapes[0]):
            ignored_expanding_layer = reshapes[0]
            converter_context.merge_descriptors(ignored_expanding_layer, descriptor)

    @classmethod
    def is_expanding_boxes_input(cls, converter_context, boxes_expand_dim_descriptor):
        boxes_expand_dim_op = boxes_expand_dim_descriptor.child_ops[-1]
        shape = converter_context.graph_helper.get_op_output_shape(boxes_expand_dim_op)
        return len(shape) == 4 and shape[2] == 1

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        if len(input_descriptors) != 2:
            raise ConverterError(get_error_message('ERROR_TF_SSD_NMS_REQUIRES_2_INPUTS'))

        input_names = []
        input_shapes = []
        for i in input_descriptors:
            tensors = converter_context.get_output_tensors_between(i, descriptor)
            if len(tensors) != 1:
                raise ConverterError(get_error_message('ERROR_TF_SSD_NMS_REQUIRES_SINGLE_INPUT_TENSOR'))

            input_shapes.append(converter_context.graph_helper.get_op_output_shape(tensors[0].op))

        for index, shape in enumerate(input_shapes):
            output_names = input_descriptors[index].output_names
            if len(shape) == 3 and shape[-1] == 4:
                input_names = output_names + input_names
            else:
                input_names.extend(output_names)

        classes_shape = converter_context.graph_helper.get_op_output_shape(descriptor.classes_output_op)

        return converter_context.model.add_multi_class_nms_layer(name=descriptor.layer_name,
                                                                 input_names=input_names,
                                                                 output_names=descriptor.output_names,
                                                                 scoreThreshold=descriptor.score_threshold,
                                                                 iouThreshold=descriptor.iou_threshold,
                                                                 maxDetectionPerClass=classes_shape[-1],
                                                                 maxTotalDetections=classes_shape[-1])
