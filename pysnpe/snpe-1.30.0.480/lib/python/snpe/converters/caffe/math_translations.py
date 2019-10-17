# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from itertools import repeat

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import math

from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from snpe.converters.common.converter_ir import op_adapter
from snpe.converters.common.utils import code_to_message, snpe_translation_utils
from snpe.converters.common.utils.snpe_converter_utils import *


# -----------------------------------------------------------------
# Converter translations
# -----------------------------------------------------------------
class CaffeDetectionOutputTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):

        if "priorbox" in converter_type(layer.type, 'caffe'):
            # save as noop op. We will retrieve this in optimizations, concat all priorboxes and
            # add it to detection_output op. After everything is added, concat nodes for priorboxes and all
            # priorboxes will be pruned in the optimization
            op = op_adapter.Noop(layer.name)
            op.priorbox_box_output = self.process_ssd_priorbox_layer(layer, graph)
            return op
        else:
            params = layer.detection_output_param if layer.type == 'DetectionOutput' else layer.ssd_detection_output_param
            nms_param = params.nms_param

            if "keep_top_k" not in str(params):
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_MISSING_SSD_PARAM')
                                 (str(layer.name), 'keep_top_k'))
            if int(params.keep_top_k) < 0:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_INVALID_SSD_PARAM')
                                 (str(layer.name), 'keep_top_k', int(params.keep_top_k)))

            # 7: [image_batch, label, confidence, x_min, y_min, x_max, y_max]
            # 0 dimension indicates dynamic resizing of # of outputs
            input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
            input_dim = graph.get_buffer(input_name).get_buf_dims()
            output_dims = [[input_dim[0], 1, 0, 7]]

            code_map = {caffe_pb2.PriorBoxParameter.CodeType.Value('CORNER'): "PRIORBOX_TYPE_CORNER",
                        caffe_pb2.PriorBoxParameter.CodeType.Value('CENTER_SIZE'): "PRIORBOX_TYPE_CENTER_SIZE",
                        caffe_pb2.PriorBoxParameter.CodeType.Value('CORNER_SIZE'): "PRIORBOX_TYPE_CORNER_SIZE"}
            code_type = code_map[params.code_type]

            return op_adapter.DetectionOutputOp(layer.name,
                                                output_dims=output_dims,
                                                num_classes=params.num_classes,
                                                share_location=params.share_location,
                                                background_label_id=params.background_label_id,
                                                nms_threshold=nms_param.nms_threshold,
                                                confidence_threshold=params.confidence_threshold,
                                                nms_top_k=nms_param.top_k,
                                                nms_eta=nms_param.eta,
                                                code_type=code_type,
                                                keep_top_k=params.keep_top_k,
                                                variance_encoded_in_target=params.variance_encoded_in_target
                                                )

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_dims]

    @staticmethod
    def process_ssd_priorbox_layer(layer, graph):
        def get_ssd_aspect_ratios(params):
            aspect_ratios_ = [1.]
            for val in params.aspect_ratio:
                ar_ = val
                already_exist = False
                for prior in aspect_ratios_:
                    if math.fabs(ar_ - prior) < 1e-6:
                        already_exist = True
                        break
                if not already_exist:
                    aspect_ratios_.append(ar_)
                    if params.flip is True:
                        aspect_ratios_.append(1. / ar_)
            return aspect_ratios_

        def get_ssd_num_priors(params):
            # OPEN_SOURCE_START
            # The following code is derived based on code from the following open source projects/packages.
            # Project name: caffe
            # Branch: ssd
            # Note: There are few minor changes to accommodate for SNPE framework

            # determine how many aspect ratios user
            # provided to calculate the number of
            # prior boxes
            aspect_ratios_ = get_ssd_aspect_ratios(params)
            num_priors_ = int(len(aspect_ratios_) * len(params.min_size))

            if "max_size" in str(params):
                for val in params.max_size:
                    num_priors_ += 1

            return num_priors_

            # OPEN_SOURCE_END

        log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERTING_LAYER')(str(layer.type), layer.name))
        input_names = graph.naming_policy.get_input_names(layer, layer.bottom)
        input_dims = [graph.get_buffer(name).get_buf_dims() for name in input_names]

        output_names = graph.naming_policy.get_output_names(layer, layer.top)
        if len(output_names) != 1:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_INVALID_SSD_PARAM')
                             (str(layer.name), 'num prior box outputs', len(layer.top)))

        # CHW format
        input_layer_height = input_dims[0][2]
        input_layer_width = input_dims[0][3]
        # get the first node in the graph(the input node) and its first output to get model input dims
        model_input_buf_dims = graph.get_output_buffers(graph.list_nodes()[0])[0].get_buf_dims()
        model_input_height = model_input_buf_dims[2]
        model_input_width = model_input_buf_dims[3]
        prior_box_params = layer.prior_box_param if layer.type == 'PriorBox' else layer.ssd_prior_box_param

        if prior_box_params.step_w == 0 or prior_box_params.step_h == 0:
            step_w = float(model_input_width) / input_layer_width
            step_h = float(model_input_height) / input_layer_height
        else:
            step_w = prior_box_params.step_w
            step_h = prior_box_params.step_h
        min_sizes = [min_size for min_size in prior_box_params.min_size]
        max_sizes = [max_size for max_size in prior_box_params.max_size]
        aspect_ratios = get_ssd_aspect_ratios(prior_box_params)
        variances = [variance for variance in prior_box_params.variance]

        num_priors = get_ssd_num_priors(prior_box_params)
        output_dim = (input_layer_height * input_layer_width * num_priors * 4)

        prior_box_output = []
        prior_box_variances = []

        for h in range(0, input_layer_height):
            for w in range(0, input_layer_width):
                center_x = (w + prior_box_params.offset) * step_w
                center_y = (h + prior_box_params.offset) * step_h

                for s in range(0, len(min_sizes)):
                    # first prior: aspect_ratio = 1, size = min_size
                    min_size = min_sizes[s]
                    box_width = box_height = min_size
                    # x_min
                    prior_box_output.append((center_x - box_width / 2.) / model_input_width)
                    # y_min
                    prior_box_output.append((center_y - box_height / 2.) / model_input_height)
                    # x_max
                    prior_box_output.append((center_x + box_width / 2.) / model_input_width)
                    # y_max
                    prior_box_output.append((center_y + box_height / 2.) / model_input_height)

                    if len(max_sizes) > 0:
                        if len(min_sizes) != len(max_sizes):
                            raise ValueError(
                                code_to_message.get_error_message('ERROR_CAFFE_INVALID_SSD_PARAM')
                                (str(layer.name), 'Number of min and max size for SsdPriorbox must be same',
                                 str(len(min_sizes)) + ',' + str(len(max_sizes))))

                        max_size = max_sizes[s]
                        # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        box_width = box_height = math.sqrt(min_size * max_size)
                        # x_min
                        prior_box_output.append((center_x - box_width / 2.) / model_input_width)
                        # y_min
                        prior_box_output.append((center_y - box_height / 2.) / model_input_height)
                        # x_max
                        prior_box_output.append((center_x + box_width / 2.) / model_input_width)
                        # y_max
                        prior_box_output.append((center_y + box_height / 2.) / model_input_height)

                    # rest of priors
                    for r in range(0, len(aspect_ratios)):
                        ar = aspect_ratios[r]
                        if math.fabs(ar - 1.) < 1e-6:
                            continue
                        box_width = min_size * math.sqrt(ar)
                        box_height = min_size / math.sqrt(ar)
                        # x_min
                        prior_box_output.append((center_x - box_width / 2.) / model_input_width)
                        # y_min
                        prior_box_output.append((center_y - box_height / 2.) / model_input_height)
                        # x_max
                        prior_box_output.append((center_x + box_width / 2.) / model_input_width)
                        # y_max
                        prior_box_output.append((center_y + box_height / 2.) / model_input_height)

        # clip the prior's coordinate such that it is within [0, 1]
        if prior_box_params.clip:
            for d in range(0, output_dim):
                prior_box_output[d] = min(max(prior_box_output[d], 0.), 1.)

        # set the variances in separate array and collectively add the end of all the priorboxes.
        # This is since we are concatenating on axis 1
        # Below is the implementation for this in caffe: top_data += top[0]->offset(0, 1);
        if len(variances) == 1:
            # implementing this as follows: caffe_set < Dtype > (output_dim, Dtype(variance_[0]), top_data);
            if variances[0] == 0:
                prior_box_variances.extend(repeat(0, output_dim))  # NOLINT(caffe / alt_fn)
            else:
                for i in range(0, output_dim):
                    prior_box_variances.append(variances[0])
        else:
            for h in range(0, input_layer_height):
                for w in range(0, input_layer_width):
                    for i in range(0, num_priors):
                        for j in range(0, 4):
                            prior_box_variances.append(variances[j])

        return [prior_box_output, prior_box_variances]


CaffeTranslations.register_translation(CaffeDetectionOutputTranslation(),
                                       converter_type('detectionoutput', 'caffe'),
                                       converter_type('ssdoutput', 'caffe'),
                                       converter_type('priorbox', 'caffe'),
                                       converter_type('ssdpriorbox', 'caffe'),
                                       op_adapter.DetectionOutputOp.TRANSLATION_KEY)


class CaffeEluTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        alpha = float(layer.elu_param.alpha)
        return op_adapter.NeuronOp(layer.name,
                                   snpe_translation_utils.extract_activation(layer.type),
                                   a=alpha)


CaffeTranslations.register_translation(CaffeEluTranslation(),
                                       converter_type('elu', 'caffe'))


class CaffeElementwiseTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        elementwise_param = layer.eltwise_param
        op = elementwise_param.operation
        if op == elementwise_param.PROD:
            return op_adapter.ElementwiseProductOp(layer.name)
        elif op == elementwise_param.SUM:
            coeffs = list(elementwise_param.coeff)
            input_names = layer.bottom
            if len(coeffs) < len(input_names):
                log_warning(code_to_message.get_warning_message('WARNING_CAFFE_FEWER_COEFFS_THAN_INPUT_NUM'))
                coeffs.extend([1.0 for _ in range(len(input_names)-len(coeffs))])
            elif len(coeffs) > len(input_names):
                log_warning(code_to_message.get_warning_message('WARNING_CAFFE_MORE_COEFFS_THAN_INPUT_NUM'))

            return op_adapter.ElementwiseSumOp(layer.name,
                                               coeffs=coeffs)
        elif op == elementwise_param.MAX:
            return op_adapter.ElementwiseMaxOp(layer.name)
        else:
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_UNRECOGNIZED_ELEMENTWISE_OP')(str(layer.name), str(op)))


CaffeTranslations.register_translation(CaffeElementwiseTranslation(),
                                       converter_type('eltwise', 'caffe'),
                                       op_adapter.ElementwiseProductOp.TRANSLATION_KEY,
                                       op_adapter.ElementwiseSumOp.TRANSLATION_KEY,
                                       op_adapter.ElementwiseMaxOp.TRANSLATION_KEY)


class CaffeLrnTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        lrn_param = layer.lrn_param
        window_size = getattr(lrn_param, "local_size")
        alpha = getattr(lrn_param, "alpha")
        beta = getattr(lrn_param, "beta")
        k = getattr(lrn_param, "k")

        norm_region = ""
        # get actual name of enum so that it is more concrete to use later
        for enum_name in lrn_param.NormRegion._enum_type.values_by_name:
            if getattr(lrn_param, enum_name) == lrn_param.norm_region:
                norm_region = enum_name

        return op_adapter.LrnOp(layer.name,
                                norm_region=norm_region,
                                window_size=window_size,
                                alpha=alpha,
                                beta=beta,
                                k=k)


CaffeTranslations.register_translation(CaffeLrnTranslation(),
                                       converter_type('lrn', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.LRN, 'caffe'),
                                       op_adapter.LrnOp.TRANSLATION_KEY)


class CaffePowerTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        power_param = layer.power_param
        scale = getattr(power_param, "scale", 1.0)
        shift = getattr(power_param, "shift", 1.0)
        power = getattr(power_param, "power", 1.0)

        return op_adapter.PowerOp(layer.name,
                                  scale=scale,
                                  shift=shift,
                                  power=power)


CaffeTranslations.register_translation(CaffePowerTranslation(),
                                       converter_type('power', 'caffe'),
                                       op_adapter.PowerOp.TRANSLATION_KEY)


class CaffePreluTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        prelu_param = layer.prelu_param
        bias = graph.weights.get_prelu_weights(layer)

        return op_adapter.PreluOp(layer.name,
                                  coeff=bias,
                                  channel_shared=prelu_param.channel_shared)


CaffeTranslations.register_translation(CaffePreluTranslation(),
                                       converter_type('prelu', 'caffe'),
                                       op_adapter.PreluOp.TRANSLATION_KEY)


class CaffeReluTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        layer_type = layer.type
        if layer_type == caffe.proto.caffe_pb2.V1LayerParameter.RELU:
            layer_type = "RELU"
        return op_adapter.NeuronOp(layer.name,
                                   snpe_translation_utils.extract_activation(layer_type))


CaffeTranslations.register_translation(CaffeReluTranslation(),
                                       converter_type('relu', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.RELU, 'caffe'),
                                       op_adapter.NeuronOp.TRANSLATION_KEY)


class CaffeSigmoidTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        layer_type = layer.type
        if layer_type == caffe.proto.caffe_pb2.V1LayerParameter.SIGMOID:
            layer_type = "SIGMOID"
        return op_adapter.NeuronOp(layer.name,
                                   snpe_translation_utils.extract_activation(layer_type),
                                   a=1.0)


CaffeTranslations.register_translation(CaffeSigmoidTranslation(),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.SIGMOID, 'caffe'),
                                       converter_type('sigmoid', 'caffe'))


class CaffeSoftmaxTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        return op_adapter.SoftmaxOp(layer.name)


CaffeTranslations.register_translation(CaffeSoftmaxTranslation(),
                                       converter_type('softmax', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.SOFTMAX, 'caffe'),
                                       op_adapter.SoftmaxOp.TRANSLATION_KEY)


class CaffeTanhTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        layer_type = layer.type
        if layer_type == caffe.proto.caffe_pb2.V1LayerParameter.TANH:
            layer_type = "TANH"
        return op_adapter.NeuronOp(layer.name,
                                   snpe_translation_utils.extract_activation(layer_type),
                                   a=1.0,
                                   b=1.0)


CaffeTranslations.register_translation(CaffeTanhTranslation(),
                                       converter_type('tanh', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.TANH, 'caffe'))
