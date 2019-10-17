# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from snpe.converters.common.converter_ir import translation, op_adapter, op_graph
from snpe.converters.common.converter_ir.axis_tracker import AxisTracker
from snpe.converters.onnx.op_schema import OpSchemaBase, OpSchemaDict, OP_SCHEMA_REGISTRY
from .util import *

OnnxTranslations = translation.TranslationBank()

# onnx specific translation method keys
ADD_INPUT_OP = "ADD_INPUT_OP"
SUPPORTED_VERSION = "SUPPORTED_VERSION"


class OnnxTranslationBase(translation.ConversionTranslationBase):
    def __init__(self):
        translation.ConversionTranslationBase.__init__(self)
        self.register_method(SUPPORTED_VERSION, self.get_supported_version)
        self._op_schema = OpSchemaDict()  # dictionary-style class that maps {version:op_schema}

    def extract_parameters(self, src_op, graph):
        raise NotImplementedError("extract_parameters for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, graph):
        return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        return list(map(str, src_op.output))

    def populate_axes_format(self, node, graph):
        output_buffers = graph.get_output_buffers(node)
        for buf in output_buffers:
            if buf.rank() == 4:
                buf.axis_format = AxisTracker.AxisFormat.NCS
            elif buf.rank() == 2:
                buf.axis_format = AxisTracker.AxisFormat.FEATURE
            else:
                buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

            # update axis_format if node is an input bases on user request
            if node.op.type == op_adapter.InputOp.TRANSLATION_KEY:
                if node.op.input_encoding_in == op_graph.InputEncodings.TIME_SERIES:
                    log_assert(buf.rank() == 3,
                               code_to_message.get_error_message("ERROR_TIMESERIES_UNEXPECTED_RANK")
                               (node.op.name, buf.rank()))
                    buf.axis_format = AxisTracker.AxisFormat.TBF
                elif node.op.input_encoding_in == op_graph.InputEncodings.OTHER:
                    buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    def get_supported_version(self):
        try:
            version = list(map(int, self._op_schema.get_schemas().keys()))
            return version
        except Exception as e:
            raise NotImplementedError("get_supported_version for {} not implemented ".format(str(self.__class__.__name__)))

    def register_op_schema(self, name, versions, unsupported_attrs=None):
        """
               Wraps Onnx's internal schema definition into a condensed op_schema_dict internal object (OpSchemaDict)
               which contains individual op_schema(s)(OpSchemaBase) that tie supported attributes,
               number of inputs and outputs to the appropriate op version

               :param name: The type of op to be registered
               :param versions : list of versions of the op to be registered. Note the versions must be available in
                                 the Onnx spec.
               :param unsupported_attrs: A list of lists of unsupported attrs, which are in the Onnx spec
                                        for an op version but are not supported by the translation

               registers the resulting op_schema dictionary with the translation, as well as with a
               global schema registry

        """

        if unsupported_attrs:
            while len(unsupported_attrs) < len(versions):
                unsupported_attrs.append(unsupported_attrs[0])
        else:
            unsupported_attrs = [[] for _ in range(len(versions))]

        for i, version in enumerate(versions):
            schema = defs.get_schema(name, version, '')
            op_schema = OpSchemaBase()
            op_schema.populate_op_schema(schema, unsupported_attrs[i])
            self._op_schema.add_schema(op_schema, version)

        OP_SCHEMA_REGISTRY[name.lower()] = self._op_schema

    def op_schema(self, version=None):
        if version is not None:
            return self._op_schema.get_schemas(version)
        values = list(self._op_schema.get_schemas().values())
        return values[-1]


# -----------------------------------------------------------------
# Converter translations
# Note: ONNX doesn't have input op(s) but we create one for the IR
# -----------------------------------------------------------------
class OnnxInputTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_method(ADD_INPUT_OP, self.add_input_op)

    def add_input_op(self, input_, graph):
        name = str(input_.name)
        tensor_shape = input_.type.tensor_type.shape
        shape = [int(dim.dim_value) for dim in tensor_shape.dim]
        neg_idx = [idx for idx in range(len(shape)) if shape[idx] < 0]

        if neg_idx:
            raise RuntimeError('SNPE does not support negative/placeholder dimensions.'
                               'Expected shape: {} > 0'.format(shape))

        node = graph.add_input(name, shape)
        self.populate_axes_format(node, graph)


OnnxTranslations.register_translation(OnnxInputTranslation(),
                                      converter_type('input', 'onnx'),
                                      op_adapter.InputOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Dropout and other Noops
# ------------------------------------------------------------------------------
class OnnxNoopTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Dropout', [1])

    def extract_parameters(self, src_op, graph):
        return op_adapter.Noop(src_op.name)

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxNoopTranslation(),
                                      converter_type('Dropout', 'onnx'),
                                      op_adapter.Noop.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   StaticOp
# ------------------------------------------------------------------------------
# 'Static' ops are transformations applied to weights, which do not produce
# an actual runtime output.
class OnnxStaticTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        return op_adapter.StaticOp(src_op.name)

    def infer_output_shapes(self, op, input_shapes):
        return []

    def get_supported_version(self):
        return {}


OnnxTranslations.register_translation(OnnxStaticTranslation(), op_adapter.StaticOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Class OpVersionInfo
# ------------------------------------------------------------------------------
# Returns name and version information about an op from a particular model
class OpVersionInfo:
    def __init__(self):
        self.model_opset_version = 0

    @staticmethod
    def update_schema_registry(src_op_type, op_version):
        """ Updates the schema registry so that get_op_schema(src_op_type) will always return the appropriate schema
            for the global model opset version """
        op_schema_dict = OP_SCHEMA_REGISTRY[src_op_type.lower()]
        op_schema_keys = list(op_schema_dict.get_schemas().keys())
        if op_schema_keys[-1] != str(op_version):
           op_schema_dict.reorder_op_schemas(str(op_version))

    def validate_op_ver(self, src_op, supported_version):
        """

        :param src_op: The op from the Onnx framework
        :param supported_version: The version of the op supported by the Onnx Converter
        :return: a warning if the opset_version for the source op does not match any version supported
                 by the converter
                 updates the schema registry if the src_op version is supported, so that any schema calls (self.op_schema()
                 or get_op_schema) will return the src_op_version.
        """

        # This uses the model version to extract the associated opset version for a given op
        # For example: The scenarios are described below
        #              supported_version = [1, 6, 7]
        #              Model_opset_version = 3,    Model_opset_version = 7,   Model_opset_version = 9
        #              current_op_version = 1,     current_op_version = 7     current_op_version = 8
        #                                                                     returns a warning
        #

        current_op_version = int(defs.C.get_schema(src_op.op_type, self.model_opset_version, '').since_version)
        if current_op_version not in supported_version:
            log_warning(code_to_message.get_warning_message("WARNING_OP_VERSION_NOT_SUPPORTED")
                        (src_op.op_type, list(map(int, supported_version)), [current_op_version]))
        else:
            self.update_schema_registry(src_op.op_type, current_op_version)

    def set_global_op_ver(self, model):
        """ Sets the highest global op version supported by the model"""
        # Get the global opset version
        if len(model.opset_import) > 1:
            log_warning(code_to_message.get_warning_message("WARNING_OPSET_VERSION"))

        for opset in model.opset_import:
            if opset.version > self.model_opset_version:
                self.model_opset_version = opset.version
