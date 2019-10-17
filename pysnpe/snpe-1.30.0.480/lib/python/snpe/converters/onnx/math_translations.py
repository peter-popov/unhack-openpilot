# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *
from snpe.converters.common.utils import snpe_translation_utils


# ------------------------------------------------------------------------------
#   Abs
# ------------------------------------------------------------------------------
class OnnxAbsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Abs', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryAbsOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxAbsTranslation(),
                                      converter_type('Abs', 'onnx'),
                                      op_adapter.ElementwiseUnaryAbsOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Add
# ------------------------------------------------------------------------------
class OnnxAddTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Add', [1, 6, 7])
        self.input_names = []

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseSumOp(str(src_op.name))

        # checks to see if op requires broadcast
        if is_broadcast(src_op):
            log_warning(code_to_message.get_warning_message("WARNING_BROADCAST_ADD"))

        # we check if the op has any const or static input, which would then be interpreted as a
        # bias input if the op is performing addition/subtraction or weights if the op is performing
        # a multiplication/division
        self.input_names, _, bias = set_to_weights_and_biases(src_op, graph, mode='bias')
        if bias is not None:
                op.bias = bias
        return op

    def extract_input_names(self, src_op, graph):
        return self.input_names


OnnxTranslations.register_translation(OnnxAddTranslation(),
                                      converter_type('Add', 'onnx'),
                                      op_adapter.ElementwiseSumOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ArgMax
# ------------------------------------------------------------------------------
class OnnxArgMaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ArgMax', [1, 7])

    def extract_parameters(self, src_op, graph):
        # these parameters belong to ArgMax
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        return op_adapter.ArgMaxOp(str(src_op.name),
                                   axis=params.axis,
                                   keepdims=params.keepdims)

    def infer_output_shapes(self, op, input_shapes):
        output_shape = []
        for axis, shape in enumerate(input_shapes[0][:]):
            if axis == op.axis:
                if op.keepdims:
                    output_shape.append(1)
                continue
            output_shape.append(shape)
        return [output_shape]


OnnxTranslations.register_translation(OnnxArgMaxTranslation(),
                                      converter_type('ArgMax', 'onnx'),
                                      op_adapter.ArgMaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Div
# ------------------------------------------------------------------------------
class OnnxDivTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Div', [1, 6, 7])
        self.input_names = []

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseDivOp(str(src_op.name))

        if is_broadcast(src_op):
            log_warning(code_to_message.get_warning_message("WARNING_BROADCAST_DIV"))

        # we check if the op has any const or static input, which would then be interpreted as a
        # bias input if the op is performing addition/subtraction or weights if the op is performing
        # a multiplication/division
        self.input_names, weights, _ = set_to_weights_and_biases(src_op, graph, mode='weights')
        if weights:
                op.weights = 1/weights
        return op

    def extract_input_names(self, src_op, graph):
        return self.input_names


OnnxTranslations.register_translation(OnnxDivTranslation(),
                                      converter_type('Div', 'onnx'),
                                      op_adapter.ElementwiseDivOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Elu
# ------------------------------------------------------------------------------
class OnnxEluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Elu', [1, 6])

    def extract_parameters(self, src_op, graph):
        # these parameters belong to Elu
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.NeuronOp(str(src_op.name),
                                   snpe_translation_utils.extract_activation(src_op.op_type),
                                   a=params.alpha)


OnnxTranslations.register_translation(OnnxEluTranslation(),
                                      converter_type('Elu', 'onnx'))


# ------------------------------------------------------------------------------
#   Exp
# ------------------------------------------------------------------------------
class OnnxExpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Exp', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryExpOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxExpTranslation(),
                                      converter_type('Exp', 'onnx'),
                                      op_adapter.ElementwiseUnaryExpOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Floor
# ------------------------------------------------------------------------------
class OnnxFloorTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Floor', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryFloorOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxFloorTranslation(),
                                      converter_type('Floor', 'onnx'),
                                      op_adapter.ElementwiseUnaryFloorOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   GEMM
# ------------------------------------------------------------------------------
class OnnxGemmTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gemm', [1])
        self._op_schema.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        log_warning(code_to_message.get_warning_message("WARNING_GEMM"))
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))
        weights, bias = graph.weights.fetch(*input_names[1:])
        weights *= params.alpha
        # for GEMM, weights are supposed to be B and thus KxN.
        # for FC, weights are supposed to be NxK and get transposed
        # implicitly. Transpose explicitly here so that they wind up as NxK
        # for axes_to_snpe_order
        weights = numpy.ascontiguousarray(numpy.transpose(weights, (1, 0)))
        bias *= params.beta
        return op_adapter.FullyConnectedOp(str(src_op.name), [weights], bias)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'transA':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)


OnnxTranslations.register_translation(OnnxGemmTranslation(), converter_type('Gemm', 'onnx'))


# ------------------------------------------------------------------------------
#   Identity
# ------------------------------------------------------------------------------
class OnnxIdentityTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Identity', [1])

    def extract_parameters(self, src_op, graph):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op. Otherwise the identity op is a no-op that
        # gets squashed later.
        if not graph.has_buffer(src_op.input[0]):
            const_input = graph.weights.fetch(str(src_op.input[0]))
            graph.weights.insert(str(src_op.output[0]), const_input)
            return op_adapter.ConstantOp(src_op.output[0], const_input)

        return op_adapter.Noop(str(src_op.name))

    def extract_input_names(self, src_op, graph):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op which do not need an input name.
        if not graph.has_buffer(src_op.input[0]):
            return []
        return str(src_op.input[0])


OnnxTranslations.register_translation(OnnxIdentityTranslation(),
                                      converter_type('Identity', 'onnx'))


# ------------------------------------------------------------------------------
#   Log
# ------------------------------------------------------------------------------
class OnnxLogTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Log', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryLogOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxLogTranslation(),
                                      converter_type('Log', 'onnx'),
                                      op_adapter.ElementwiseUnaryLogOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Matmul
# ------------------------------------------------------------------------------
class OnnxMatMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MatMul', [1])

    def extract_parameters(self, src_op, graph):
        log_warning(code_to_message.get_warning_message("WARNING_MATMUL"))
        input_names = list(map(str, src_op.input))
        # SNPE currently only supports FC, so given AxB, B MUST be a set of
        # static weights
        weights = graph.weights.fetch(input_names[1])
        bias = numpy.zeros(weights.shape[1], dtype=numpy.float32)
        return op_adapter.FullyConnectedOp(str(src_op.name), [weights], bias)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]


OnnxTranslations.register_translation(OnnxMatMulTranslation(), converter_type('MatMul', 'onnx'))


# ------------------------------------------------------------------------------
#   Max
# ------------------------------------------------------------------------------
class OnnxMaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Max', [1, 6, 8])

    def extract_parameters(self, src_op, graph):
        assert_no_broadcast(src_op)
        return op_adapter.ElementwiseMaxOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxMaxTranslation(),
                                      converter_type('Max', 'onnx'),
                                      op_adapter.ElementwiseMaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Mul
# ------------------------------------------------------------------------------
class OnnxMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mul', [1, 6, 7])
        self.input_names = []

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseProductOp(str(src_op.name))

        # check if one of the inputs requires broadcasting
        # if Weights is provided as one of the inputs. This is determined if one
        # of the inputs is a const

        if is_broadcast(src_op):
            log_warning(code_to_message.get_warning_message("WARNING_BROADCAST_MUL"))

        # we check if the op has any const or static input, which would then be interpreted as a
        # bias input if the op is performing addition/subtraction or weights if the op is performing
        # a multiplication/division
        self.input_names, weights, _ = set_to_weights_and_biases(src_op, graph, mode='weights')
        if weights is not None:
                op.weights = weights
        return op

    def extract_input_names(self, src_op, graph):
        return self.input_names


OnnxTranslations.register_translation(OnnxMulTranslation(),
                                      converter_type('Mul', 'onnx'),
                                      op_adapter.ElementwiseProductOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Neg
# ------------------------------------------------------------------------------
class OnnxNegTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Neg', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnaryNegOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxNegTranslation(),
                                      converter_type('Neg', 'onnx'),
                                      op_adapter.ElementwiseUnaryNegOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ReduceMax
# ------------------------------------------------------------------------------
class OnnxReduceMaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ReduceMax', [1, 6])

    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(str(src_op.input[0]))
        schema = self.op_schema()
        schema.replace_default_values(axes=range(input_buf.rank()))
        params = extract_attributes(src_op, schema=schema)

        return op_adapter.ReduceMaxOp(str(src_op.name),
                                      axes=params.axes,
                                      keepdims=params.keepdims)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        output_shape = []
        for axis, shape in enumerate(input_shapes[0][:]):
            if axis in op.axes:
                if op.keepdims:
                    output_shape.append(1)
                continue
            output_shape.append(shape)
        return [output_shape]


OnnxTranslations.register_translation(OnnxReduceMaxTranslation(),
                                      converter_type('ReduceMax', 'onnx'),
                                      op_adapter.ReduceMaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   ReduceSum
# ------------------------------------------------------------------------------
class OnnxReduceSumTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ReduceSum', [1, 6])

    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(src_op.input[0])
        schema = self.op_schema()
        schema.replace_default_values(axes=range(input_buf.rank()))
        params = extract_attributes(src_op, schema=schema)

        return op_adapter.ReduceSumOp(str(src_op.name),
                                      axes=params.axes,
                                      keepdims=params.keepdims)

    def extract_input_names(self, src_op, graph):
        return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        output_shape = []
        for axis, shape in enumerate(input_shapes[0][:]):
            if axis in op.axes:
                if op.keepdims:
                    output_shape.append(1)
                continue
            output_shape.append(shape)
        return [output_shape]


OnnxTranslations.register_translation(OnnxReduceSumTranslation(),
                                      converter_type('ReduceSum', 'onnx'),
                                      op_adapter.ReduceSumOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class OnnxReluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Relu', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(str(src_op.name), snpe_translation_utils.extract_activation(src_op.op_type))


OnnxTranslations.register_translation(OnnxReluTranslation(),
                                      converter_type('Relu', 'onnx'),
                                      op_adapter.NeuronOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sigmoid
# ------------------------------------------------------------------------------
class OnnxSigmoidTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sigmoid', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.NeuronOp(str(src_op.name), snpe_translation_utils.extract_activation(src_op.op_type), a=1.0)


OnnxTranslations.register_translation(OnnxSigmoidTranslation(), converter_type('Sigmoid', 'onnx'))


# ------------------------------------------------------------------------------
#   Sin
# ------------------------------------------------------------------------------
class OnnxSinTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sin', [7])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnarySinOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxSinTranslation(),
                                      converter_type('Sin', 'onnx'),
                                      op_adapter.ElementwiseUnarySinOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class OnnxSoftmaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Softmax', [1])
        self._op_schema.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(str(src_op.input[0]))

        # TO-D0: this function call exists only to validate that axis=1 since
        # SNPE does not support any other case. Make this more generic
        extract_attributes(src_op, schema=self.op_schema(), validate=True)

        log_assert(input_buf.rank() == 2,
                   "Node %s: SNPE supports softmax only for inputs of rank 2",
                   str(src_op.name))
        return op_adapter.SoftmaxOp(str(src_op.name))

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'axis':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)


OnnxTranslations.register_translation(OnnxSoftmaxTranslation(),
                                      converter_type('Softmax', 'onnx'),
                                      op_adapter.SoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sub
# ------------------------------------------------------------------------------
class OnnxSubTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sub', [1, 6, 7])
        self.input_names = []

    def extract_parameters(self, src_op, graph):
        op = op_adapter.ElementwiseSubOp(str(src_op.name))

        # checks to see if op requires broadcast
        if is_broadcast(src_op):
            log_warning(code_to_message.get_warning_message("WARNING_BROADCAST_SUB"))

        # we check if the op has any const or static input, which would then be interpreted as a
        # bias input if the op is performing addition/subtraction or weights if the op is performing
        # a multiplication/division.
        self.input_names, _, bias = set_to_weights_and_biases(src_op, graph, mode='bias')
        if bias is not None:
            op.bias = -1*bias

        return op

    def extract_input_names(self, src_op, graph):
        return self.input_names


OnnxTranslations.register_translation(OnnxSubTranslation(),
                                      converter_type('Sub', 'onnx'),
                                      op_adapter.ElementwiseSubOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Sum
# ------------------------------------------------------------------------------
class OnnxSumTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sum', [1, 6, 7])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseSumOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxSumTranslation(), converter_type('Sum', 'onnx'))


# ------------------------------------------------------------------------------
#   Sqrt
# ------------------------------------------------------------------------------
class OnnxSqrtTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sqrt', [1, 6])

    def extract_parameters(self, src_op, graph):
        return op_adapter.ElementwiseUnarySqrtOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxSqrtTranslation(),
                                      converter_type('Sqrt', 'onnx'),
                                      op_adapter.ElementwiseUnarySqrtOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Tanh, ScaledTanh
# ------------------------------------------------------------------------------
class OnnxTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tanh', [1, 6])
        self.register_op_schema('ScaledTanh', [1, 6])

    def extract_parameters(self, src_op, graph):
        # these parameters belong to ScaledTanh
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.NeuronOp(str(src_op.name),
                                   snpe_translation_utils.extract_activation(src_op.op_type),
                                   a=params.alpha,
                                   b=params.beta)


OnnxTranslations.register_translation(OnnxTanhTranslation(),
                                      converter_type('Tanh', 'onnx'),
                                      converter_type('ScaledTanh', 'onnx'))
