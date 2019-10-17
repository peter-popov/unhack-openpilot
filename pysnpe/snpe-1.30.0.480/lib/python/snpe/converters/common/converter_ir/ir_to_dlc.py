# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys

from snpe.converters.common.utils import code_to_message

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

from snpe.converters.common.converter_ir import translation, op_adapter
from snpe.converters.common.utils import code_to_message
from snpe.converters.common.utils.snpe_converter_utils import *


# ------------------------------------------------------------------------------
#   Module Level enum/Functions
# ------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# IR consts to dlc dictionary. This holds the translation between the string constants in IR graph
# to what is defined in modeltools.
# -------------------------------------------------------------------------------------------------
ir_consts_to_dlc = {
    # conv
    "PADDING_ZERO": modeltools.PADDING_ZERO,
    "PADDING_REFLECT": modeltools.PADDING_REFLECT,
    "PADDING_CONSTANT": modeltools.PADDING_CONSTANT,
    "PADDING_SIZE_EXPLICIT": modeltools.PADDING_SIZE_EXPLICIT,
    "PADDING_SIZE_IMPLICIT_VALID": modeltools.PADDING_SIZE_IMPLICIT_VALID,
    "PADDING_SIZE_IMPLICIT_SAME": modeltools.PADDING_SIZE_IMPLICIT_SAME,
    "PADDING_SIZE_EXPLICIT_FLOOR": modeltools.PADDING_SIZE_EXPLICIT_FLOOR,
    "PADDING_SIZE_EXPLICIT_ASYMMETRIC": modeltools.PADDING_SIZE_EXPLICIT_ASYMMETRIC,

    "NEURON_RELU": modeltools.NEURON_RELU,
    "NEURON_RELU_MIN_MAX": modeltools.NEURON_RELU_MIN_MAX,
    "NEURON_TANH": modeltools.NEURON_TANH,
    "NEURON_LOGISTIC": modeltools.NEURON_LOGISTIC,
    "NEURON_ELU": modeltools.NEURON_ELU,
    "NEURON_NONE": modeltools.NEURON_NONE,

    # pooling
    "POOL_MAX": modeltools.POOL_MAX,
    "POOL_AVG": modeltools.POOL_AVG,

    # scaling
    "RESIZE_BILINEAR": modeltools.RESIZE_BILINEAR,
    "RESIZE_NEAREST_NEIGHBOR": modeltools.RESIZE_NEAREST_NEIGHBOR,

    # ssd
    "PRIORBOX_TYPE_CORNER": modeltools.PRIORBOX_TYPE_CORNER,
    "PRIORBOX_TYPE_CENTER_SIZE": modeltools.PRIORBOX_TYPE_CENTER_SIZE,
    "PRIORBOX_TYPE_CORNER_SIZE": modeltools.PRIORBOX_TYPE_CORNER_SIZE,

    # embedding
    "EMBEDDING_PARTITION_STRATEGY_MOD": modeltools.EMBEDDING_PARTITION_STRATEGY_MOD,
    "EMBEDDING_PARTITION_STRATEGY_DIV": modeltools.EMBEDDING_PARTITION_STRATEGY_DIV,

    # channel shuffle
    "CHANNEL_SHUFFLE_GROUPED": modeltools.CHANNEL_SHUFFLE_GROUPED,

    # layer affinity
    "LAYER_AFFINITY_CPU_FLOAT32": modeltools.LAYER_AFFINITY_CPU_FLOAT32,
    "LAYER_AFFINITY_GPU_FLOAT32_16_HYBRID": modeltools.LAYER_AFFINITY_GPU_FLOAT32_16_HYBRID,
    "LAYER_AFFINITY_DSP_FIXED8_TF": modeltools.LAYER_AFFINITY_DSP_FIXED8_TF,
    "LAYER_AFFINITY_GPU_FLOAT16": modeltools.LAYER_AFFINITY_GPU_FLOAT16,
}

# translation method keys
IR_TO_DLC = 'ir_to_dlc'

DlcTranslations = translation.TranslationBank()


def save(graph, converter):

    # get converter args for saving dlc
    output_path = converter.output_model_path if converter.output_model_path else converter.input_model_path + '.dlc'
    converter_command = getattr(converter, "converter_command", "")
    copyright_str = getattr(converter, "copyright_str", "")
    validation_target = getattr(converter, "validation_target", [])
    enable_strict_validation = getattr(converter, "enable_strict_validation", False)

    model = modeltools.Model()

    # add validation target
    if len(validation_target) == 0:
        log_debug3("no validation target specified. Using defaults.")
        model.add_validation_targets(model.get_validation_targets())
    else:
        log_debug3("validation target :" + str(tuple(validation_target)))
        model.add_validation_targets(tuple(validation_target))

    # set validation mode
    if enable_strict_validation:
        log_debug3("strict validation is enabled.")
        model.set_strict_validation(True)

    log_info(code_to_message.get_progress_message("INFO_DLC_SAVE_LOCATION")(output_path))
    DlcTranslations.apply_method_to_all_ops(IR_TO_DLC, graph, model)

    for buf in graph.list_buffers():
        model.set_buffer_axis_order(buf.name, buf.get_axis_order())
    if graph.quantization_params:
        model.add_quantization_params(graph.quantization_params)
    model.set_converter_command(converter_command)
    model.set_model_copyright(copyright_str)
    model.save(output_path)


# ------------------------------------------------------------------------------
#   Translations
# ------------------------------------------------------------------------------
def register(dlc_translation):
    DlcTranslations.register_translation(dlc_translation(), dlc_translation.TARGET)
    return dlc_translation


class DlcTranslationBase(translation.Translation):
    def __init__(self):
        translation.Translation.__init__(self)
        self.register_method(IR_TO_DLC, self.add_ir_node_as_dlc_layer)

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        raise NotImplementedError("add_ir_node_as_dlc_layer for {} not implemented ".
                                  format(str(self.__class__.__name__)))


@register
class DlcInputTranslation(DlcTranslationBase):
    TARGET = op_adapter.InputOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_data_layer(node.op.name,
                             node.op.shape,
                             node.op.input_encoding_in,
                             node.op.input_encoding_out,
                             node.op.input_type)


@register
class DlcArgMaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.ArgMaxOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_argmax_layer(node.op.name,
                               node.input_names[0],
                               node.output_names[0],
                               node.op.axis,
                               node.op.keepdims)


@register
class DlcBatchnormTranslation(DlcTranslationBase):
    TARGET = op_adapter.BatchnormOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_batchnorm_layer(node.op.name,
                                  node.op.weights,
                                  node.op.bias,
                                  node.op.compute_statistics,
                                  node.op.use_mu_sigma,
                                  node.op.across_spatial,
                                  node.input_names[0],
                                  node.output_names[0],
                                  node.op.epsilon)


@register
class DlcChannelShuffleTranslation(DlcTranslationBase):
    TARGET = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_channel_shuffle_layer(node.op.name,
                                        node.op.groups,
                                        ir_consts_to_dlc[node.op.shuffle_mode],
                                        node.input_names[0],
                                        node.output_names[0])


@register
class DlcConvolutionTranslation(DlcTranslationBase):
    TARGET = op_adapter.ConvolutionOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_conv_layer(node.op.name,
                             node.op.weights,
                             node.op.bias,
                             node.op.padx,
                             node.op.pady,
                             ir_consts_to_dlc[node.op.padding_mode],
                             ir_consts_to_dlc[node.op.padding_size_strategy],
                             node.op.stridex,
                             node.op.stridey,
                             node.op.dilationx,
                             node.op.dilationy,
                             node.input_names[0],
                             node.output_names[0],
                             node.op.groups)


@register
class DlcConcatTranslation(DlcTranslationBase):
    TARGET = op_adapter.ConcatOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):

        if node.op.axis > 4:
            raise ValueError(code_to_message.get_error_message('ERROR_SNPE_TILE_AXIS_NOT_SUPPORTED')
                             (str(node.op.name), node.op.axis))

        model.add_concatenation_layer(node.op.name,
                                      node.input_names,
                                      node.output_names[0],
                                      node.op.axis)


@register
class DlcConstantTranslation(DlcTranslationBase):
    TARGET = op_adapter.ConstantOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        node.op.name = node.output_names[0]
        model.add_const_layer(node.op.name,
                              list(node.op.tensor.shape),
                              node.op.tensor,
                              node.op.quantizable)


@register
class DlcCropTranslation(DlcTranslationBase):
    TARGET = op_adapter.CropOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_crop_layer(node.op.name,
                             node.op.offsets,
                             node.op.output_shape,
                             node.input_names[0],
                             node.output_names[0])


@register
class DlcCrossCorrelationTranslation(DlcTranslationBase):
    TARGET = op_adapter.CrossCorrelationOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        log_assert(len(node.input_names) == 2, "Layer %s: expected exactly two input blobs" % node.op.name)
        model.add_cross_correlation_layer(node.op.name,
                                          node.input_names[0],
                                          node.input_names[1],
                                          node.output_names[0])


@register
class DlcDeconvolutionTranslation(DlcTranslationBase):
    TARGET = op_adapter.DeconvolutionOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_deconvolution_layer(node.op.name,
                                      node.op.weights,
                                      node.op.bias,
                                      node.op.stride,
                                      int(ir_consts_to_dlc[node.op.padding_size_strategy]),
                                      node.op.padx,
                                      node.op.pady,
                                      node.input_names[0],
                                      node.output_names[0],
                                      node.op.output_width,
                                      node.op.output_height,
                                      node.op.groups)


@register
class DlcDetectionOutputTranslation(DlcTranslationBase):
    TARGET = op_adapter.DetectionOutputOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_ssd_detection_output_layer(node.op.name,
                                             node.input_names,
                                             node.output_names,
                                             node.op.output_dims,
                                             node.op.num_classes,
                                             node.op.share_location,
                                             node.op.background_label_id,
                                             node.op.nms_threshold,
                                             node.op.nms_top_k,
                                             node.op.nms_eta,
                                             ir_consts_to_dlc[node.op.code_type],
                                             node.op.priorbox_data,
                                             node.op.keep_top_k,
                                             node.op.variance_encoded_in_target,
                                             node.op.confidence_threshold
                                             )


@register
class DlcDropoutTranslation(DlcTranslationBase):
    TARGET = op_adapter.DropoutOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_dropout_layer(node.op.name,
                                node.op.keep,
                                node.input_names[0],
                                node.output_names[0])


@register
class DlcElementwiseDivTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseDivOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_div_layer(node.op.name,
                                        node.input_names,
                                        node.output_names[0])


@register
class DlcElementwiseMaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseMaxOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_max_layer(node.op.name,
                                        node.input_names,
                                        node.output_names[0])


@register
class DlcElementwiseProductTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseProductOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_product_layer(node.op.name,
                                            node.input_names,
                                            node.output_names[0])


@register
class DlcElementwiseSumTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseSumOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        coeffs = node.op.coeffs[:]
        num_missing_coeffs = len(node.input_names) - len(coeffs)
        if num_missing_coeffs > 0:
            coeffs.extend([1.0] * num_missing_coeffs)
        model.add_elementwise_sum_layer(node.op.name,
                                        coeffs,
                                        node.input_names,
                                        node.output_names[0])


@register
class DlcElementwiseUnaryAbsTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryAbsOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_unary_abs_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0])


@register
class DlcElementwiseUnaryExpTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryExpOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_unary_exp_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0])


@register
class DlcElementwiseUnaryFloorTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryFloorOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_unary_floor_layer(node.op.name,
                                                node.input_names[0],
                                                node.output_names[0])


@register
class DlcElementwiseUnaryLogTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryLogOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_unary_log_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0])


@register
class DlcElementwiseUnaryNegTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseUnaryNegOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_unary_neg_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0])


@register
class DlcElementwiseUnarySinTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseUnarySinOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_unary_sin_layer(node.op.name,
                                              node.input_names[0],
                                              node.output_names[0])


@register
class DlcElementwiseSubTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseSubOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_sub_layer(node.op.name,
                                        node.input_names,
                                        node.output_names[0])


@register
class DlcElementwiseUnarySqrtTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseUnarySqrtOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_elementwise_unary_sqrt_layer(node.op.name,
                                               node.input_names[0],
                                               node.output_names[0])


@register
class DlcFullyConnectedTranslation(DlcTranslationBase):
    TARGET = op_adapter.FullyConnectedOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_fc_layer(node.op.name,
                           node.op.weights_list,
                           node.op.bias,
                           node.input_names,
                           node.output_names[0])


@register
class DlcGatherTranslation(DlcTranslationBase):
    TARGET = op_adapter.GatherOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_gather_layer(node.op.name,
                               node.input_names[0],
                               node.input_names[1],
                               node.output_names[0],
                               node.op.axis)


@register
class DlcGenerateProposalsOp(DlcTranslationBase):
    TARGET = op_adapter.GenerateProposalsOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_generate_proposals_layer(node.op.name,
                                           node.op.spatial_scale,
                                           node.op.pre_nms_top_n,
                                           node.op.post_nms_top_n,
                                           node.op.nms_thresh,
                                           node.op.min_size,
                                           node.op.correct_transform_coords,
                                           node.op.anchors,
                                           node.op.im_info,
                                           node.input_names[0],
                                           node.input_names[1],
                                           node.output_names[0],
                                           node.ouput_names[1])


@register
class DlcGruTranslation(DlcTranslationBase):
    TARGET = op_adapter.GruOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_gru_layer(node.op.name,
                            node.op.state_gate,
                            node.op.forget_gate,
                            node.op.control_gate,
                            ir_consts_to_dlc[node.op.activation],
                            ir_consts_to_dlc[node.op.gate_activation],
                            ir_consts_to_dlc[node.op.rec_gate_activation],
                            node.op.backwards,
                            node.input_names[0],
                            node.output_names[0])


@register
class DlcLrnTranslation(DlcTranslationBase):
    TARGET = op_adapter.LrnOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        if node.op.norm_region == "ACROSS_CHANNELS":
            model.add_cmrn_layer(node.op.name,
                                 node.op.window_size,
                                 float(node.op.alpha/node.op.window_size),
                                 node.op.beta,
                                 node.op.k,
                                 node.input_names[0],
                                 node.output_names[0])
        else:
            model.add_local_norm_layer(node.op.name,
                                       node.op.window_size,
                                       node.op.alpha,
                                       node.op.beta,
                                       node.op.k,
                                       node.input_names[0],
                                       node.output_names[0])


@register
class DlcLstmTranslation(DlcTranslationBase):
    TARGET = op_adapter.LstmOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        input_name = node.input_names[0]
        model.add_lstm_layer(node.op.name,
                             node.op.gate_weights,
                             node.op.gate_bias,
                             node.op.recurrent_weights,
                             node.op.w_xc_static,
                             node.op.backward,
                             node.op.reset_state_at_time_step_0,
                             input_name,
                             node.op.sequence_continuation_name,
                             node.op.x_static_name,
                             node.op.c_0_input_name,
                             node.op.h_0_input_name,
                             node.output_names)


@register
class DlcMaxYTranslation(DlcTranslationBase):
    TARGET = op_adapter.MaxYOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_max_y_layer(node.op.name,
                              node.input_names[0],
                              node.output_names[0])


@register
class DlcNeuronTranslation(DlcTranslationBase):
    TARGET = op_adapter.NeuronOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_neuron_layer(node.op.name,
                               ir_consts_to_dlc[node.op.neuron_type],
                               node.input_names[0],
                               node.output_names[0],
                               node.op.a,
                               node.op.b,
                               node.op.min_clamp,
                               node.op.max_clamp)


@register
class DlcPadTranslation(DlcTranslationBase):
    TARGET = op_adapter.PadOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        supported_modes = {'constant': ir_consts_to_dlc['PADDING_CONSTANT'],
                           'reflect': ir_consts_to_dlc['PADDING_REFLECT']}
        node.op.mode = supported_modes[node.op.mode]
        model.add_pad_layer(node.op.name,
                            node.input_names[0],
                            node.op.pads,
                            node.op.mode,
                            node.op.constant_value,
                            node.output_names[0])


@register
class DlcPoolTranslation(DlcTranslationBase):
    TARGET = op_adapter.PoolOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_pooling_layer(node.op.name,
                                ir_consts_to_dlc[node.op.pool_type],
                                node.op.size_x,
                                node.op.size_y,
                                node.op.stride_x,
                                node.op.stride_y,
                                node.op.pad_x,
                                node.op.pad_y,
                                ir_consts_to_dlc[node.op.padding_size_strategy],
                                node.input_names[0],
                                node.output_names[0],
                                node.op.pool_region_include_padding)


@register
class DlcPermuteTranslation(DlcTranslationBase):
    TARGET = op_adapter.PermuteOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_permute_layer(node.op.name,
                                node.op.order,
                                node.input_names[0],
                                node.output_names[0])


@register
class DlcPowerTranslation(DlcTranslationBase):
    TARGET = op_adapter.PowerOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_power_layer(node.op.name,
                              node.op.scale,
                              node.op.shift,
                              node.op.power,
                              node.input_names[0],
                              node.output_names[0])


@register
class DlcPreluTranslation(DlcTranslationBase):
    TARGET = op_adapter.PreluOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        if node.op.channel_shared:
            raise ValueError(code_to_message.get_error_message('ERROR_PRELU_NON_CHANNEL_SHARED_SUPPORT_ONLY')
                             (str(node.op.name)))

        model.add_prelu_layer(node.op.name,
                              node.op.coeff,
                              node.input_names[0],
                              node.output_names[0])


@register
class DlcProposalTranslation(DlcTranslationBase):
    TARGET = op_adapter.ProposalOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_proposal_layer(node.op.name,
                                 node.op.feat_stride,
                                 node.op.scales,
                                 node.op.ratios,
                                 node.op.anchor_base_size,
                                 node.op.min_bbox_size,
                                 node.op.max_num_proposals,
                                 node.op.max_num_rois,
                                 node.op.iou_threshold_nms,
                                 node.input_names,
                                 node.output_names[0])


@register
class DlcReduceMaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.ReduceMaxOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_reduction_max_layer(node.op.name,
                                      node.input_names[0],
                                      node.output_names[0],
                                      node.op.axes,
                                      node.op.keepdims)


@register
class DlcReduceSumTranslation(DlcTranslationBase):
    TARGET = op_adapter.ReduceSumOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_reduction_sum_layer(node.op.name,
                                      node.input_names[0],
                                      node.output_names[0],
                                      node.op.axes,
                                      node.op.keepdims)


@register
class DlcReshapeTranslation(DlcTranslationBase):
    TARGET = op_adapter.ReshapeOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_reshape_layer(node.op.name,
                                node.op.output_shape,
                                node.input_names[0],
                                node.output_names[0])


@register
class DlcRNormTranslation(DlcTranslationBase):
    TARGET = op_adapter.RNormOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        if node.op.across_channels:
            add_method = model.add_cmrn_layer
        else:
            add_method = model.add_local_norm_layer

        add_method(node.op.name,
                   node.op.size,
                   node.op.alpha,
                   node.op.beta,
                   node.op.k,
                   node.input_names[0],
                   node.output_names[0])


@register
class DlcRoiAlignTranslation(DlcTranslationBase):
    TARGET = op_adapter.RoiAlignOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_roialign_layer(node.op.name,
                                 node.op.spatial_scale,
                                 node.op.pooled_size_h,
                                 node.op.pooled_size_w,
                                 node.op.sampling_ratio,
                                 node.input_names[0],
                                 node.input_names[1],
                                 node.output_names[0],
                                 node.output_names[1] if len(node.output_names) > 1 else "",
                                 node.op.tiled_batch_h,
                                 node.op.tiled_batch_w,
                                 node.op.batch_pad_h,
                                 node.op.batch_pad_w,
                                 node.op.pad_value)


@register
class DlcRoiPoolingTranslation(DlcTranslationBase):
    TARGET = op_adapter.RoiPoolingOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        log_assert(node.op.output_shape[0] == 1,
                   code_to_message.get_error_message("ERROR_ROI_POOL_BATCH_UNSUPPORTED"))

        model.add_roipooling_layer(node.op.name,
                                   node.op.pooled_size_w,
                                   node.op.pooled_size_h,
                                   node.op.spatial_scale,
                                   node.op.output_shape,
                                   node.input_names,
                                   node.output_names[0])


@register
class DlcResizeTranslation(DlcTranslationBase):
    TARGET = op_adapter.ResizeOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        supported_modes = {'nearest': ir_consts_to_dlc['RESIZE_NEAREST_NEIGHBOR'],
                           # for now mapping linear to bilinear since pytorch bilinear is
                           # changing to linear when model gets exported to onnx.
                           'linear': ir_consts_to_dlc['RESIZE_BILINEAR'],
                           'bilinear': ir_consts_to_dlc['RESIZE_BILINEAR']}
        node.op.resize_mode = supported_modes[node.op.resize_mode]
        model.add_scaling_layer(node.op.name,
                                node.op.output_shape,
                                node.op.pad_value,
                                node.op.maintain_aspect_ratio,
                                node.op.resize_mode,
                                node.op.scale_height,
                                node.op.scale_width,
                                node.input_names[0],
                                node.output_names[0],
                                node.op.align_corners)


@register
class DlcRnnTransformationTranslation(DlcTranslationBase):
    TARGET = op_adapter.RnnTransformationOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_tx_layer(node.op.name,
                           node.op.weights,
                           node.op.bias,
                           node.op.activation,
                           node.input_names[0],
                           node.output_names[0])


@register
class DlcScaleTranslation(DlcTranslationBase):
    TARGET = op_adapter.ScaleOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_scale_layer(node.op.name,
                              node.op.weights,
                              node.op.bias,
                              node.input_names,
                              node.output_names[0],
                              node.op.axis,
                              node.op.num_axes,)


@register
class DlcSliceTranslation(DlcTranslationBase):
    TARGET = op_adapter.SliceOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_slice_layer(node.op.name,
                              node.input_names[0],
                              node.op.axis,
                              node.op.slice_points,
                              node.output_names)


@register
class DlcSoftmaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_softmax_layer(node.op.name,
                                node.input_names[0],
                                node.output_names[0])


@register
class DlcSubtractMeanTranslation(DlcTranslationBase):
    TARGET = op_adapter.SubtractMeanOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_subtract_mean_layer(node.op.name,
                                      node.op.mean_values,
                                      node.input_names[0],
                                      node.output_names[0])


@register
class DlcUdlTranslation(DlcTranslationBase):
    TARGET = op_adapter.UdlOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_user_defined_layer(node.op.name,
                                     node.op.layer_type,
                                     node.input_names,
                                     node.output_names,
                                     node.op.output_dims,
                                     node.op.blob)


@register
class DlcUpsampleIndexBaseTranslation(DlcTranslationBase):
    TARGET = op_adapter.UpsampleIndexBasedOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        pool_id = model.get_layer_id(node.input_names[1])
        model.add_upsample_layer(node.op.name,
                                 node.op.pool_size,
                                 node.op.pool_stride,
                                 node.op.pad,
                                 node.op.output_height,
                                 node.op.output_width,
                                 node.input_names[0],
                                 node.output_names[0],
                                 pool_id)


@register
class DlcUpsampleSparseTranslation(DlcTranslationBase):
    TARGET = op_adapter.UpsampleSparseOp.TRANSLATION_KEY

    def add_ir_node_as_dlc_layer(self, node, graph, model):
        model.add_upsample_layer(node.op.name,
                                 node.op.pool_size,
                                 node.op.pool_stride,
                                 node.op.pad,
                                 node.op.output_height,
                                 node.op.output_width,
                                 node.input_names[0],
                                 node.output_names[0])
