# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class Op(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.attrs = {}

    def addattr(self, key, source, default):
        self.attrs[key] = source.get(key, default)

    def assertattr(self, key, source):
        if key in source:
            self.attrs[key] = source[key]
        else:
            raise KeyError("Op %s missing required argument %s" % (self.name, key))

    def __getitem__(self, key):
        return self.attrs[key]

    def __setitem__(self, key, value):
        self.attrs[key] = value

    def __getattr__(self, name):
        try:
            return self.attrs[name]
        except KeyError:
            raise KeyError("op %s has no attribute %s" % (self.name, name))


class InputOp(Op):
    TRANSLATION_KEY = 'input'

    def __init__(self, name, shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.shape = shape
        self.assertattr('input_encoding_in', kargs)
        self.assertattr('input_encoding_out', kargs)
        self.assertattr('input_type', kargs)


class ArgMaxOp(Op):
    TRANSLATION_KEY = 'argmax'

    def __init__(self, name, axis, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis = axis
        self.addattr('keepdims', kargs, False)


class BatchnormOp(Op):
    TRANSLATION_KEY = 'batchnorm'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.addattr('compute_statistics', kargs, False)
        self.addattr('use_mu_sigma', kargs, False)
        self.addattr('across_spatial', kargs, False)
        self.addattr('epsilon', kargs, 1e-9)
        self.addattr('gamma', kargs, [])
        self.addattr('beta', kargs, [])


class ChannelShuffleOp(Op):
    TRANSLATION_KEY = 'channel_shuffle'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('groups', kargs)
        self.addattr('groups', kargs, None)
        self.addattr('shuffle_mode', kargs, "CHANNEL_SHUFFLE_GROUPED")


class ConvolutionOp(Op):
    TRANSLATION_KEY = 'convolution'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.assertattr('padx', kargs)
        self.assertattr('pady', kargs)
        self.assertattr('stridex', kargs)
        self.assertattr('stridey', kargs)
        self.assertattr('dilationx', kargs)
        self.assertattr('dilationy', kargs)
        self.addattr('groups', kargs, 1)
        self.addattr('padding_mode', kargs, "PADDING_ZERO")
        self.addattr('padding_size_strategy', kargs, "PADDING_SIZE_EXPLICIT")


class ConcatOp(Op):
    TRANSLATION_KEY = 'concatenation'

    def __init__(self, name, axis):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis = axis


class ConstantOp(Op):
    TRANSLATION_KEY = 'constant'

    def __init__(self, name, tensor, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.tensor = tensor
        self.addattr('quantizable', kargs, True)


class CropOp(Op):
    TRANSLATION_KEY = 'crop'

    def __init__(self, name, offsets, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.offsets = offsets
        self.output_shape = output_shape


class CrossCorrelationOp(Op):
    TRANSLATION_KEY = 'cross_correlation'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class DeconvolutionOp(Op):
    TRANSLATION_KEY = 'deconvolution'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.addattr('stride', kargs, 1)
        self.addattr('padx', kargs, 0)
        self.addattr('pady', kargs, 0)
        self.addattr('padding_size_strategy', kargs, "PADDING_SIZE_EXPLICIT")
        self.addattr('output_paddingx', kargs, 0)
        self.addattr('output_paddingy', kargs, 0)
        self.assertattr('output_height', kargs)
        self.assertattr('output_width', kargs)
        self.addattr('groups', kargs, 1)


class DetectionOutputOp(Op):
    TRANSLATION_KEY = 'detection_output'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('output_dims', kargs)
        self.assertattr('num_classes', kargs)
        self.assertattr('share_location', kargs)
        self.assertattr('background_label_id', kargs)
        self.assertattr('nms_threshold', kargs)
        self.assertattr('confidence_threshold', kargs)
        self.assertattr('nms_top_k', kargs)
        self.assertattr('nms_eta', kargs)
        self.assertattr('code_type', kargs)
        self.assertattr('keep_top_k', kargs)
        self.assertattr('variance_encoded_in_target', kargs)
        self.addattr('priorbox_data', kargs, None)  # gets filled out in optimization


class DropoutOp(Op):
    TRANSLATION_KEY = 'dropout'

    def __init__(self, name, keep):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.keep = keep


class ElementwiseDivOp(Op):
    TRANSLATION_KEY = 'elementwise_div'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseUnaryAbsOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_abs'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseUnaryFloorOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_floor'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseUnaryExpOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_exp'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseUnaryLogOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_log'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseMaxOp(Op):
    TRANSLATION_KEY = 'elementwise_max'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseUnaryNegOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_neg'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseProductOp(Op):
    TRANSLATION_KEY = 'elementwise_product'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseSumOp(Op):
    TRANSLATION_KEY = 'elementwise_sum'

    def __init__(self, name, coeffs =[]):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.coeffs = coeffs


class ElementwiseSubOp(Op):

    TRANSLATION_KEY = 'elementwise_sub'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('scale_input', kargs, [])


class ElementwiseUnarySinOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_sin'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class ElementwiseUnarySqrtOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_sqrt'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class FullyConnectedOp(Op):
    TRANSLATION_KEY = 'fully_connected'

    def __init__(self, name, weights_list, bias):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights_list = weights_list
        self.bias = bias


class GatherOp(Op):
    TRANSLATION_KEY = 'gather'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('axis', kargs, 0)


class GenerateProposalsOp(Op):
    TRANSLATION_KEY = 'generate_proposals'

    def __init__(self, name, anchors, im_info, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pre_nms_top_n', kargs)
        self.assertattr('post_nms_top_n', kargs)
        self.assertattr('nms_thresh', kargs)
        self.assertattr('min_size', kargs)
        self.addattr('correct_transform_coords', kargs, True)


class GruOp(Op):
    TRANSLATION_KEY = 'gru'

    def __init__(self, name, state_gate, forget_gate, control_gate, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.state_gate = state_gate
        self.forget_gate = forget_gate
        self.control_gate = control_gate
        self.addattr('activation', kargs, "NEURON_LOGISTIC")
        self.addattr('gate_activation', kargs, "NEURON_LOGISTIC")
        self.addattr('rec_gate_activation', kargs, "NEURON_TANH")
        self.addattr('backwards', kargs, False)


class LrnOp(Op):
    TRANSLATION_KEY = 'lrn'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('norm_region', kargs, "")
        self.assertattr('window_size', kargs)
        self.assertattr('alpha', kargs)
        self.assertattr('beta', kargs)
        self.assertattr('k', kargs)


class LstmOp(Op):
    TRANSLATION_KEY = 'lstm'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('gate_weights', kargs)
        self.assertattr('gate_bias', kargs)
        self.assertattr('recurrent_weights', kargs)
        self.addattr('w_xc_static', kargs, None)
        self.addattr('backward', kargs, False)
        self.addattr('activations', kargs, [])
        self.addattr('reset_state_at_time_step_0', kargs, False)
        self.addattr('h_0_input_name', kargs, '')
        self.addattr('c_0_input_name', kargs, '')
        self.addattr('sequence_continuation_name', kargs, '')
        self.addattr('x_static_name', kargs, '')
        self.addattr('w_cc', kargs, None)
        self.addattr('cell_clip', kargs, 0.0)
        self.addattr('w_p', kargs, None)
        self.addattr('b_p', kargs, None)
        self.addattr('projection_clip', kargs, 0.0)
        self.addattr('w_n', kargs, 0.0)
        self.addattr('epsilon', kargs, 0.0)



class MaxYOp(Op):
    TRANSLATION_KEY = 'max_y'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class NegOp(Op):
    TRANSLATION_KEY = 'neg'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class NeuronOp(Op):
    TRANSLATION_KEY = 'neuron'

    def __init__(self, name, neuron_type, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.neuron_type = neuron_type
        self.addattr('a', kargs, 0.0)
        self.addattr('b', kargs, 0.0)
        self.addattr('min_clamp', kargs, 0.0)
        self.addattr('max_clamp', kargs, 0.0)


class Noop(Op):
    TRANSLATION_KEY = 'noop'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class PoolOp(Op):
    TRANSLATION_KEY = 'pool'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_type', kargs)
        self.assertattr('size_x', kargs)
        self.assertattr('size_y', kargs)
        self.addattr('stride_x', kargs, 1)
        self.addattr('stride_y', kargs, 1)
        self.addattr('pad_x', kargs, 0)
        self.addattr('pad_y', kargs, 0)
        self.addattr('padding_size_strategy', kargs, "PADDING_SIZE_EXPLICIT")
        self.addattr('pool_region_include_padding', kargs, True)


class PadOp(Op):
    TRANSLATION_KEY = 'pad'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pads', kargs)
        self.addattr('mode', kargs, "PADDING_CONSTANT")
        self.addattr('constant_value', kargs, 0)


class PermuteOp(Op):
    TRANSLATION_KEY = 'permute'

    def __init__(self, name, order):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.order = order


class PowerOp(Op):
    TRANSLATION_KEY = 'power'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('scale', kargs, 1.0)
        self.addattr('shift', kargs, 0.0)
        self.addattr('power', kargs, 1.0)


class PreluOp(Op):
    TRANSLATION_KEY = 'prelu'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('coeff', kargs)
        self.addattr('channel_shared', kargs, False)


class ProposalOp(Op):
    TRANSLATION_KEY = 'proposal'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('feat_stride', kargs)
        self.assertattr('scales', kargs)
        self.assertattr('ratios', kargs)
        self.assertattr('anchor_base_size', kargs)
        self.assertattr('min_bbox_size', kargs)
        self.assertattr('max_num_proposals', kargs)
        self.assertattr('max_num_rois', kargs)
        self.assertattr('iou_threshold_nms', kargs)


class ReduceMaxOp(Op):
    TRANSLATION_KEY = 'reduce_max'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kargs)
        self.addattr('keepdims', kargs, True)


class ReduceSumOp(Op):
    TRANSLATION_KEY = 'reduce_sum'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kargs)
        self.addattr('keepdims', kargs, True)


class ReshapeOp(Op):
    TRANSLATION_KEY = 'reshape'

    def __init__(self, name, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_shape = output_shape


class RNormOp(Op):
    TRANSLATION_KEY = 'rnorm'

    def __init__(self, name, size, alpha, beta, k, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.addattr('across_channels', kargs, True)


class RoiAlignOp(Op):
    TRANSLATION_KEY = 'roi_align'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('sampling_ratio', kargs)
        # implode batch parameters
        self.addattr('tiled_batch_h', kargs, -1)
        self.addattr('tiled_batch_w', kargs, -1)
        self.addattr('batch_pad_h', kargs, -1)
        self.addattr('batch_pad_w', kargs, -1)
        self.addattr('pad_value', kargs, 0.0)


class RoiPoolingOp(Op):
    TRANSLATION_KEY = 'roi_pooling'

    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('spatial_scale', kargs)
        self.output_shape = output_shape


class ResizeOp(Op):
    TRANSLATION_KEY = 'resize'

    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_shape = output_shape
        self.addattr('pad_value', kargs, 0.0)
        self.addattr('maintain_aspect_ratio', kargs, False)
        self.addattr('resize_mode', kargs, "RESIZE_BILINEAR")
        self.addattr('scale_height', kargs, 0.0)
        self.addattr('scale_width', kargs, 0.0)
        self.addattr('align_corners', kargs, False)


class RnnTransformationOp(Op):
    TRANSLATION_KEY = 'rnn_transformation'

    def __init__(self, name, weights, bias, activation):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.activation = activation


class ScaleOp(Op):
    TRANSLATION_KEY = 'scale'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.assertattr('axis', kargs)
        self.assertattr('num_axes', kargs)


class SliceOp(Op):
    TRANSLATION_KEY = 'slice'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axis', kargs)
        self.assertattr('slice_points', kargs)
        self.addattr('output_shape', kargs, [])


class StaticOp(Op):
    TRANSLATION_KEY = 'static'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class SoftmaxOp(Op):
    TRANSLATION_KEY = 'softmax'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)


class SubtractMeanOp(Op):
    TRANSLATION_KEY = 'subtract_mean'

    def __init__(self, name, mean_values):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.mean_values = mean_values

class UdlOp(Op):
    TRANSLATION_KEY = 'udl'

    def __init__(self, name, layer_type, blob, output_dims, expected_input_axis_orders, expected_output_axis_orders):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.layer_type = layer_type
        self.blob = blob
        self.output_dims = output_dims
        self.expected_input_axis_orders = expected_input_axis_orders
        self.expected_output_axis_orders = expected_output_axis_orders


class Upsample(ResizeOp):
    TRANSLATION_KEY = "upsample"


class UpsampleIndexBasedOp(Op):
    TRANSLATION_KEY = 'upsample_index_based'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_size', kargs)
        self.addattr('pool_stride', kargs, 1)
        self.addattr('pad', kargs, 0)
        self.addattr('output_height', kargs, -1)
        self.addattr('output_width', kargs, -1)


class UpsampleSparseOp(Op):
    TRANSLATION_KEY = 'upsample_sparse'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_size', kargs)
        self.addattr('pool_stride', kargs, 1)
        self.addattr('pad', kargs, 0)
        self.addattr('output_height', kargs, -1)
        self.addattr('output_width', kargs, -1)
