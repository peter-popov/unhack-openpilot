# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from math import ceil, floor

from .onnx_translations import *
from snpe.converters.common.utils import snpe_translation_utils


# ------------------------------------------------------------------------------
#   AveragePool, MaxPool
# ------------------------------------------------------------------------------
class OnnxPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('AveragePool', [1])
        self.register_op_schema('MaxPool', [1])
        self._op_schema.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        padding_size_strategy = extract_padding_mode(params.auto_pad, src_op.name)
        if pads_righthanded(params.pads):
            padding_size_strategy = "PADDING_SIZE_EXPLICIT_ASYMMETRIC"
        if str(src_op.op_type) == 'AveragePool':
            pool_type = "POOL_AVG"
        else:
            pool_type = "POOL_MAX"

        return op_adapter.PoolOp(src_op.name,
                                 pool_type=pool_type,
                                 size_y=params.kernel_shape[0],
                                 size_x=params.kernel_shape[1],
                                 stride_y=params.strides[0],
                                 stride_x=params.strides[1],
                                 pad_y=params.pads[2],
                                 pad_x=params.pads[3],
                                 padding_size_strategy=padding_size_strategy,
                                 pool_region_include_padding=False)

    def infer_output_shapes(self, op, input_shapes):
        return snpe_translation_utils.get_pool_output_shape(op, input_shapes)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'pads':
            if not (pads_righthanded(attr_value) or pads_symmetric(attr_value)):
                raise ValueError(code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))


OnnxTranslations.register_translation(OnnxPoolTranslation(),
                                      converter_type('AveragePool', 'onnx'),
                                      converter_type('MaxPool', 'onnx'),
                                      op_adapter.PoolOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   BatchNormalization
# ------------------------------------------------------------------------------
class OnnxBatchNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('BatchNormalization', [1, 6, 7])
        self._op_schema.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        input_names = list(src_op.input)
        gamma, beta, mu, var = graph.weights.fetch(*input_names[1:])
        # y = gamma*( (x-mu)/sqrt(var+epsilon) ) + beta
        # weights = gamma/sqrt(var+epsilon)
        weights = gamma/numpy.sqrt(var+params.epsilon)
        # bias = -mu*gamma/sqrt(var+epsilon) + beta = -mu*weights + beta
        bias = -mu*weights + beta

        return op_adapter.BatchnormOp(src_op.name,
                                      weights,
                                      bias,
                                      across_spatial=bool(params.spatial),
                                      gamma=gamma,
                                      beta=beta)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    @staticmethod
    def validate_attribute_values(self, attr_name, attr_value):
        # is_test is only supported in test mode, which is_test = 1
        if attr_name == 'is_test':
            log_assert(attr_value, code_to_message.get_error_message('ERROR_BATCHNORM_TEST_ONLY'))


OnnxTranslations.register_translation(OnnxBatchNormalizationTranslation(),
                                      converter_type('BatchNormalization', 'onnx'),
                                      op_adapter.BatchnormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Conv
# ------------------------------------------------------------------------------
class OnnxConvTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Conv', [1])
        self._op_schema.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))

        weights = graph.weights.fetch(input_names[1])

        if len(input_names) > 2:
            bias = graph.weights.fetch(input_names[2])
        else:
            input_buf = graph.get_buffer(input_names[0])
            bias = numpy.zeros(weights.shape[0], dtype=numpy.float32)

        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        if params.kernel_shape:
            log_assert(tuple(params.kernel_shape) == weights.shape[2:],
                       code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))

        padding_mode = extract_padding_mode(params.auto_pad, src_op.name)

        return op_adapter.ConvolutionOp(src_op.name,
                                        weights,
                                        bias,
                                        padx=params.pads[1],
                                        pady=params.pads[0],
                                        padding_size_strategy=padding_mode,
                                        stridex=params.strides[1],
                                        stridey=params.strides[0],
                                        dilationx=params.dilations[1],
                                        dilationy=params.dilations[0],
                                        groups=params.group)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    def infer_output_shapes(self, op, input_shapes):
        return snpe_translation_utils.get_conv_output_shape(op, input_shapes)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'pads':
            log_assert(pads_symmetric(attr_value),
                       code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))


OnnxTranslations.register_translation(OnnxConvTranslation(),
                                      converter_type('Conv', 'onnx'),
                                      op_adapter.ConvolutionOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ConvTranspose
# ------------------------------------------------------------------------------
class OnnxConvTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ConvTranspose', [1])
        self._op_schema.register_method(self.validate_attribute_values)
        self._op_schema.replace_default_values(output_shape=[0, 0], output_padding=[])

    def add_op(self, src_op, graph):
        ops = [self.extract_parameters(src_op, graph)]
        input_names = [self.extract_input_names(src_op, graph)]
        output_names = [self.extract_output_names(src_op, graph)]
        output_padding = [ops[0].output_paddingx, ops[0].output_paddingy]

        # If there is any output padding provided, the deconvolution layer becomes deconv + pad layer where
        # the pad values are incorporated into a pad layer as the number of zero paddings to add to the output
        # of the original deconv layer.
        if any(output_padding):
            log_assert(output_padding < [ops[0].stride]*2,
                       code_to_message.get_error_message("ERROR_DECONV_OUTPUT_PADDING_NOT_LESS_THAN_STRIDE")
                       (ops[0].stride, output_padding))
            input_buf = graph.get_buffer(str(src_op.input[0]))
            rank = len(input_buf.shape)
            pad_pairs = []
            for _ in range(rank):
                pad_pairs.append([0, 0])
            pad_pairs[3][1] = output_padding[1]
            pad_pairs[2][1] = output_padding[0]

            ops.append(op_adapter.PadOp(str(src_op.op_type).lower() + '_output_pad',
                                        mode='constant',
                                        pads=pad_pairs))

            # set all buffers correctly before the pad layer is added
            output_names.append(output_names[0])
            output_names[0] = 'deconv_padding_out'
            input_names.append(output_names[0])
        for i, op in enumerate(ops):
                node = graph.add(op, input_names[i], output_names[i])
                self.populate_axes_format(node, graph)

    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        weights = graph.weights.fetch(input_names[1])
        if len(input_names) > 2:
            bias = graph.weights.fetch(input_names[2])
        else:
            input_buf = graph.get_buffer(input_names[0])
            bias = numpy.zeros(weights.shape[1], dtype=numpy.float32)  # take the second dim because Onnx weights for
                                                                       # convtranspose is CMHW
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        padding_mode = extract_padding_mode(params.auto_pad, src_op.name)

        if params.kernel_shape:
            log_assert(tuple(params.kernel_shape) == weights.shape[2:],
                       code_to_message.get_error_message("ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS"))

        op = op_adapter.DeconvolutionOp(src_op.name,
                                        weights,
                                        bias,
                                        stride=params.strides[0],
                                        padding=params.pads[0],
                                        output_paddingx=params.output_padding[1] if params.output_padding else None,
                                        output_paddingy=params.output_padding[0] if params.output_padding else None,
                                        padding_size_strategy=padding_mode,
                                        output_height=params.output_shape[0],
                                        output_width=params.output_shape[1],
                                        groups=params.group)
        return op

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    def infer_output_shapes(self, op, input_shapes):
        return snpe_translation_utils.get_deconv_output_shape(op, input_shapes)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'output_padding':
            log_assert(len(attr_value) <= 2,
                       code_to_message.get_error_message("ERROR_DECONV_OUTPUT_PADDING_LENGTH_UNSUPPORTED")
                       (len(attr_value)))
        if attr_name == 'pads':
            log_assert(pads_symmetric(attr_value),
                       code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))
        elif attr_name == 'strides':
            log_assert(attr_value[0] == attr_value[1],
                       code_to_message.get_error_message("ERROR_DECONV_RECTANGULAR_STRIDE_UNSUPPORTED"))


OnnxTranslations.register_translation(OnnxConvTransposeTranslation(),
                                      converter_type('ConvTranspose', 'onnx'),
                                      op_adapter.DeconvolutionOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   FC
# ------------------------------------------------------------------------------
class OnnxFCTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        # Note: Schema is not used here since this op is not part of the Onnx spec.
        params = extract_attributes(src_op, attr_infos=
                                    [('axis', 'i', 1),
                                    ('axis_w', 'i', 1)])
        log_assert(params.axis == 1, code_to_message.get_error_message("ERROR_FC_AXIS_UNSUPPORTED"))
        log_assert(params.axis_w == 1, code_to_message.get_error_message("ERROR_FC_AXIS_W_UNSUPPORTED"))

        input_names = graph.get_input_names(src_op)
        weights, bias = graph.weights.fetch(*input_names[1:3])
        return op_adapter.FullyConnectedOp(src_op.name, [weights], bias)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        N = op.weights_list[0].shape[1]
        M = input_shapes[0][0]
        return [[M, N]]


OnnxTranslations.register_translation(OnnxFCTranslation(),
                                      converter_type('FC', 'onnx'),
                                      op_adapter.FullyConnectedOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GlobalAveragePool, GlobalMaxPool
# ------------------------------------------------------------------------------
class OnnxGlobalPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GlobalAveragePool', [1])
        self.register_op_schema('GlobalMaxPool', [1])

    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(str(src_op.input[0]))

        if str(src_op.op_type) == 'GlobalAveragePool':
            pool_type = "POOL_AVG"
        else:
            pool_type = "POOL_MAX"

        return op_adapter.PoolOp(src_op.name,
                                 pool_type=pool_type,
                                 size_x=input_buf.shape[3],
                                 size_y=input_buf.shape[2],
                                 stride_x=input_buf.shape[3],
                                 stride_y=input_buf.shape[2])


OnnxTranslations.register_translation(OnnxGlobalPoolTranslation(),
                                      converter_type('GlobalAveragePool', 'onnx'),
                                      converter_type('GlobalMaxPool', 'onnx'))


# ------------------------------------------------------------------------
#   InstanceNormalization
# ------------------------------------------------------------------------------
class OnnxInstanceNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('InstanceNormalization', [1])

    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        weights, bias = graph.weights.fetch(*input_names[1:])
        return op_adapter.BatchnormOp(src_op.name,
                                      weights,
                                      bias,
                                      compute_statistics=True,
                                      use_mu_sigma=True,
                                      across_spatial=True)
    # rest is handled by OnnxBatchNormalizationTranslation


# ------------------------------------------------------------------------------
#   MaxRoiPool
# ------------------------------------------------------------------------------
class OnnxMaxRoiPoolTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MaxRoiPool', [1])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])
        roi_buf = graph.get_buffer(input_names[1])
        output_shape = [ roi_buf.shape[0],
                         input_buf.shape[1],
                         params.pooled_shape[0],
                         params.pooled_shape[1] ]

        return op_adapter.RoiPoolingOp(src_op.name,
                                       output_shape,
                                       pooled_size_h=params.pooled_shape[0],
                                       pooled_size_w=params.pooled_shape[1],
                                       spatial_scale=params.spatial_scale)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxMaxRoiPoolTranslation(),
                                      converter_type('MaxRoiPool', 'onnx'),
                                      op_adapter.RoiPoolingOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Prelu, LeakyRelu
# ------------------------------------------------------------------------------
# Also handles LeakyRelu as a bonus.
class OnnxPreluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('PRelu', [1, 6, 7])
        self.register_op_schema('LeakyRelu', [1, 6, 7])

    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        input_buf = graph.get_buffer(input_names[0])

        if str(src_op.op_type) == 'LeakyRelu':
            params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
            bias = numpy.ones(input_buf.shape[1], dtype=numpy.float32)
            bias *= params.alpha
        else:
            slope = graph.weights.fetch(input_names[1])
            if len(slope) == 1:
                bias = numpy.ones(input_buf.shape[1], dtype=numpy.float32)
                bias *= slope[0]
            else:
                bias = numpy.require(slope, dtype=numpy.float32)

        return op_adapter.PreluOp(src_op.name, coeff=bias.tolist())

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxPreluTranslation(),
                                      converter_type('Prelu', 'onnx'),
                                      converter_type('LeakyRelu', 'onnx'),
                                      op_adapter.PreluOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Lrn
# ------------------------------------------------------------------------------
class OnnxLrnTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LRN', [1])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.RNormOp(src_op.name,
                                  params.size,
                                  params.alpha / params.size,
                                  params.beta,
                                  params.bias,
                                  across_channels=True)


OnnxTranslations.register_translation(OnnxLrnTranslation(),
                                      converter_type('LRN', 'onnx'),
                                      op_adapter.RNormOp.TRANSLATION_KEY)

