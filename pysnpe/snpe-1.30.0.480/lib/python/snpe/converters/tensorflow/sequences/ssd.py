# =============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)

box_decoder_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('Postprocessor/Tile', ['Tile']),
    ConverterSequenceNode('Postprocessor/Reshape_1', ['Reshape']),
    ConverterSequenceNode('Postprocessor/Reshape', ['Reshape']),
    ConverterSequenceNode('Postprocessor/Decode/transpose', ['Transpose']),
    ConverterSequenceNode('Postprocessor/Decode/div_3', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/div_2', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/get_center_coordinates_and_sizes/transpose', ['Transpose']),
    ConverterSequenceNode('Postprocessor/Decode/unstack', ['Unpack']),
    ConverterSequenceNode('Postprocessor/Decode/Exp', ['Exp']),
    ConverterSequenceNode('Postprocessor/Decode/get_center_coordinates_and_sizes/div_1', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/get_center_coordinates_and_sizes/sub', ['Sub']),
    ConverterSequenceNode('Postprocessor/Decode/div_1', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/Exp_1', ['Exp']),
    ConverterSequenceNode('Postprocessor/Decode/get_center_coordinates_and_sizes/div', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/get_center_coordinates_and_sizes/unstack', ['Unpack']),
    ConverterSequenceNode('Postprocessor/Decode/get_center_coordinates_and_sizes/sub_1', ['Sub']),
    ConverterSequenceNode('Postprocessor/Decode/div', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/mul', ['Mul']),
    ConverterSequenceNode('Postprocessor/Decode/get_center_coordinates_and_sizes/add_1', ['Add']),
    ConverterSequenceNode('Postprocessor/Decode/mul_3', ['Mul']),
    ConverterSequenceNode('Postprocessor/Decode/mul_1', ['Mul']),
    ConverterSequenceNode('Postprocessor/Decode/get_center_coordinates_and_sizes/add', ['Add']),
    ConverterSequenceNode('Postprocessor/Decode/mul_2', ['Mul']),
    ConverterSequenceNode('Postprocessor/Decode/div_7', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/div_6', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/div_5', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/add_1', ['Add']),
    ConverterSequenceNode('Postprocessor/Decode/div_4', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/Decode/add', ['Add']),
    ConverterSequenceNode('Postprocessor/Decode/add_3', ['Add']),
    ConverterSequenceNode('Postprocessor/Decode/add_2', ['Add']),
    ConverterSequenceNode('Postprocessor/Decode/sub_1', ['Sub']),
    ConverterSequenceNode('Postprocessor/Decode/sub', ['Sub']),
    ConverterSequenceNode('Postprocessor/Decode/stack', ['Pack']),
    ConverterSequenceNode('Postprocessor/Decode/transpose_1', ['Transpose']),
    NonConsumableConverterSequenceNode('stub_35', ['?']),
    NonConsumableConverterSequenceNode('stub_36', ['?']),
    NonConsumableConverterSequenceNode('stub_37', ['?']),
    NonConsumableConverterSequenceNode('stub_38', ['?']),
    NonConsumableConverterSequenceNode('stub_39', ['?']),
    NonConsumableConverterSequenceNode('stub_40', ['?']),
    NonConsumableConverterSequenceNode('stub_41', ['?']),
    NonConsumableConverterSequenceNode('stub_42', ['?']),
    NonConsumableConverterSequenceNode('stub_43', ['?']),
    NonConsumableConverterSequenceNode('stub_44', ['?']),
    NonConsumableConverterSequenceNode('stub_45', ['?']),
    NonConsumableConverterSequenceNode('stub_46', ['?']),
    NonConsumableConverterSequenceNode('stub_47', ['?']),
    NonConsumableConverterSequenceNode('stub_48', ['?']),
    NonConsumableConverterSequenceNode('stub_49', ['?']),
    NonConsumableConverterSequenceNode('stub_50', ['?']),
    NonConsumableConverterSequenceNode('stub_51', ['?']),
    NonConsumableConverterSequenceNode('stub_52', ['?']),
])
box_decoder_sequence.set_inputs('Postprocessor/Decode/add_3', ['Postprocessor/Decode/add_1', 'Postprocessor/Decode/div_7'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/mul_3', ['Postprocessor/Decode/div_1', 'Postprocessor/Decode/get_center_coordinates_and_sizes/sub'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/add_2', ['Postprocessor/Decode/add', 'Postprocessor/Decode/div_6'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/div_6', ['Postprocessor/Decode/mul_1', 'stub_49'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/div_3', ['Postprocessor/Decode/unstack', 'stub_41'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/sub_1', ['Postprocessor/Decode/add_1', 'Postprocessor/Decode/div_5'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/sub', ['Postprocessor/Decode/add', 'Postprocessor/Decode/div_4'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/unstack', ['Postprocessor/Decode/transpose'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/stack', ['Postprocessor/Decode/sub', 'Postprocessor/Decode/sub_1', 'Postprocessor/Decode/add_2', 'Postprocessor/Decode/add_3'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/transpose_1', ['Postprocessor/Decode/stack', 'stub_52'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/div_5', ['Postprocessor/Decode/mul', 'stub_50'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/div', ['Postprocessor/Decode/unstack', 'stub_47'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/Exp', ['Postprocessor/Decode/div_3'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/get_center_coordinates_and_sizes/add_1', ['Postprocessor/Decode/get_center_coordinates_and_sizes/unstack', 'Postprocessor/Decode/get_center_coordinates_and_sizes/div_1'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/get_center_coordinates_and_sizes/div', ['Postprocessor/Decode/get_center_coordinates_and_sizes/sub_1', 'stub_46'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/div_2', ['Postprocessor/Decode/unstack', 'stub_42'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/get_center_coordinates_and_sizes/sub_1', ['Postprocessor/Decode/get_center_coordinates_and_sizes/unstack', 'Postprocessor/Decode/get_center_coordinates_and_sizes/unstack'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/mul', ['Postprocessor/Decode/Exp', 'Postprocessor/Decode/get_center_coordinates_and_sizes/sub'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/get_center_coordinates_and_sizes/transpose', ['Postprocessor/Reshape', 'stub_43'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/div_1', ['Postprocessor/Decode/unstack', 'stub_45'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/mul_2', ['Postprocessor/Decode/div', 'Postprocessor/Decode/get_center_coordinates_and_sizes/sub_1'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/mul_1', ['Postprocessor/Decode/Exp_1', 'Postprocessor/Decode/get_center_coordinates_and_sizes/sub_1'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/transpose', ['Postprocessor/Reshape_1', 'stub_40'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/get_center_coordinates_and_sizes/add', ['Postprocessor/Decode/get_center_coordinates_and_sizes/unstack', 'Postprocessor/Decode/get_center_coordinates_and_sizes/div'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/div_7', ['Postprocessor/Decode/mul', 'stub_48'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/get_center_coordinates_and_sizes/sub', ['Postprocessor/Decode/get_center_coordinates_and_sizes/unstack', 'Postprocessor/Decode/get_center_coordinates_and_sizes/unstack'])
box_decoder_sequence.set_inputs('Postprocessor/Tile', ['stub_35', 'stub_36'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/get_center_coordinates_and_sizes/div_1', ['Postprocessor/Decode/get_center_coordinates_and_sizes/sub', 'stub_44'])
box_decoder_sequence.set_inputs('Postprocessor/Reshape_1', ['stub_37', 'stub_38'])
box_decoder_sequence.set_inputs('Postprocessor/Reshape', ['Postprocessor/Tile', 'stub_39'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/Exp_1', ['Postprocessor/Decode/div_2'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/get_center_coordinates_and_sizes/unstack', ['Postprocessor/Decode/get_center_coordinates_and_sizes/transpose'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/add_1', ['Postprocessor/Decode/mul_3', 'Postprocessor/Decode/get_center_coordinates_and_sizes/add_1'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/div_4', ['Postprocessor/Decode/mul_1', 'stub_51'])
box_decoder_sequence.set_inputs('Postprocessor/Decode/add', ['Postprocessor/Decode/mul_2', 'Postprocessor/Decode/get_center_coordinates_and_sizes/add'])
box_decoder_sequence.set_outputs(['Postprocessor/Decode/transpose_1'])


nms_sequence = GraphSequence([
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/TensorArraySizeV3', ['TensorArraySizeV3']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArraySizeV3', ['TensorArraySizeV3']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArraySizeV3', ['TensorArraySizeV3']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/range', ['Range']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_6', ['TensorArrayV3']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/range', ['Range']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_5', ['TensorArrayV3']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/range', ['Range']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_4', ['TensorArrayV3']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/TensorArrayGatherV3', ['TensorArrayGatherV3']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3', ['TensorArrayGatherV3']),
    ConverterSequenceNode('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArrayGatherV3', ['TensorArrayGatherV3']),
    NonConsumableConverterSequenceNode('add_6', ['Add']),
    ConverterSequenceNode('detection_scores', ['Identity']),
    ConverterSequenceNode('detection_boxes', ['Identity']),
    NonConsumableConverterSequenceNode('stub_15', ['?']),
    NonConsumableConverterSequenceNode('stub_16', ['?']),
    NonConsumableConverterSequenceNode('stub_17', ['?']),
    NonConsumableConverterSequenceNode('stub_18', ['?']),
    NonConsumableConverterSequenceNode('stub_19', ['?']),
    NonConsumableConverterSequenceNode('stub_20', ['?']),
    NonConsumableConverterSequenceNode('stub_21', ['?']),
    NonConsumableConverterSequenceNode('stub_22', ['?']),
    NonConsumableConverterSequenceNode('stub_23', ['?']),
    NonConsumableConverterSequenceNode('stub_24', ['?']),
    NonConsumableConverterSequenceNode('stub_25', ['?']),
])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_4', ['stub_20'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArrayGatherV3', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_4','Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/range','stub_17'])
nms_sequence.set_inputs('detection_boxes', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArrayGatherV3'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/range', ['stub_18','Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/TensorArraySizeV3','stub_19'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArraySizeV3', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_5','stub_16'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_5','Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/range','stub_16'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_6', ['stub_20'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/TensorArraySizeV3', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_6','stub_15'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/range', ['stub_21','Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArraySizeV3','stub_22'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/TensorArrayGatherV3', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_6','Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/range','stub_15'])
nms_sequence.set_inputs('add_6', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/TensorArrayGatherV3','stub_25'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_5', ['stub_20'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/range', ['stub_23','Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArraySizeV3','stub_24'])
nms_sequence.set_inputs('Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArraySizeV3', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArray_4','stub_17'])
nms_sequence.set_inputs('detection_scores', ['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3'])
# do not retrieve detection_scores and detection_classes as that cause high memory usage when finding root_candidate_assignments, specifically the itertools.product. This happens when model has a lot of identity ops.
nms_sequence.set_outputs(['Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArrayGatherV3','Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3','Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_2/TensorArrayGatherV3','add_6'])
