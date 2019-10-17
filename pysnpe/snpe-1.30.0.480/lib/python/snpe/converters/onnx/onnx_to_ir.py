# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
import traceback

from snpe.converters.common.utils import code_to_message

try:
    import onnx
except ImportError:
    raise Exception(code_to_message.get_error_message("ERROR_ONNX_NOT_FOUND")(str(sys.path)))

from snpe.converters.common.converter_ir import op_graph_optimizations, op_policies, translation
from snpe.converters.common.utils.converter_base import ConverterBase
from .util import *
from . import onnx_translations


# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class OnnxConverter(ConverterBase):
    class ArgParser(ConverterBase.ArgParser):
        def __init__(self):
            super(OnnxConverter.ArgParser, self).__init__("onnx")
            # add command-line options custom to onnx converter
            self.parser.add_optional_argument("--model_path", "-m",
                                              help="Path to the source ONNX model. "
                                                   "Note: this option is DEPRECATED, please use --input_network or -i")
            self.parser.add_optional_argument("--dry_run", type=str, nargs='?', const='info', default=None,
                                              help='Evaluates the model without actually converting any ops, and '
                                                   'returns unsupported ops/attributes as well as unused inputs and/or '
                                                   'outputs if any. Leave empty or specify "info" to see dry run as a '
                                                   'table, or specify "debug" to show more detailed messages only"')

    def __init__(self, args):
        super(OnnxConverter, self).__init__(args,
                                            naming_policy=OnnxNamePolicy(),
                                            shape_inference_policy=OnnxShapeInferencePolicy())
        self.translations = onnx_translations.OnnxTranslations
        # TODO: remove for 1.31.0
        if args.model_path:
            log_warning("Option: '--model_path', '-m' is DEPRECATED and will be removed in upcoming release."
                        " Please use '--input_network', '-i")
            self.input_model_path = args.model_path
        self.dry_run = args.dry_run
        self.op_info = onnx_translations.OpVersionInfo()

    def evaluate(self, model):
        """
        Performs a dry-run of the Onnx Model without actually converting it, highlighting potential issues with
        attributes, inputs/outputs or opset versions.
        :param model: An Onnx model
        :return:
        """
        from snpe.converters.onnx import model_evaluator
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            log_warning("Potential errors found in {} as per Onnx's in-built checker tool". format(self.input_model_path))
            log_warning("{}: {}".format(type(e), e))
        log_info('Proceeding with model evaluation...................................\n')
        model_evaluator.setup_dry_run(model, self.dry_run)

    def convert(self):
        model = onnx.load(self.input_model_path)
        self.op_info.set_global_op_ver(model)

        if self.dry_run:
            self.evaluate(model)
            sys.exit(0)

        self.graph.weights = WeightProvider(model)
        # extract inputs
        parameter_names = set()
        for tensor in model.graph.initializer:
            parameter_names.add(str(tensor.name))

        for value_info in model.graph.input:
            name = str(value_info.name)
            if name in parameter_names:
                # weights are usually listed as inputs too.
                continue
            self.translations.apply_method_to_op(converter_type("input", "onnx"),
                                                 onnx_translations.ADD_INPUT_OP, value_info, self.graph)

        # extract parameters, infer shapes, etc.
        for i, src_op in enumerate(model.graph.node):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, src_op.op_type))
            src_type = converter_type(src_op.op_type, "onnx")
            try:
                supported_version = self.translations.apply_method_to_op(src_type,
                                                                         onnx_translations.SUPPORTED_VERSION)
                self.op_info.validate_op_ver(src_op, supported_version)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

            try:
                self.translations.apply_method_to_op(src_type,
                                                     translation.ADD_OP,
                                                     src_op,
                                                     self.graph)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

        return self.graph

    def ir_optimize(self, graph, **kwargs):
        try:
            # apply graph transformations
            op_graph_optimizations.apply_graph_optimizations(graph, self.disable_batchnorm_folding, **kwargs)
            return graph
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            log_error(str(e))
            sys.exit(-1)


# ------------------------------------------------------------------------------
#   Policies
# ------------------------------------------------------------------------------
class OnnxNamePolicy(op_policies.ConversionNamePolicy):
    def __init__(self):
        op_policies.ConversionNamePolicy.__init__(self)

    def get_op_name(self, op):
        if op.name:
            return str(op.name)
        else:
            count = self.type_count.get(op.type, 0)
            self.type_count[op.type] = count+1
            return "%s_%d" % (op.type, count)


class OnnxShapeInferencePolicy(op_policies.ConversionShapeInferencePolicy):

    def infer_shape(self, op, input_shapes):
        return onnx_translations.OnnxTranslations.apply_method_to_op(op.type,
                                                                     translation.INFER_SHAPE,
                                                                     op,
                                                                     input_shapes)
