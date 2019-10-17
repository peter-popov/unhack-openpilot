# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from snpe.converters.common.converter_ir import translation, op_adapter, op_graph
from snpe.converters.common.converter_ir.axis_tracker import AxisTracker
from snpe.converters.common.utils import code_to_message
from snpe.converters.common.utils.snpe_converter_utils import *

# ------------------------------------------------------------------------------
#   CaffeTranslation
# ------------------------------------------------------------------------------
CaffeTranslations = translation.TranslationBank()


class CaffeTranslationBase(translation.ConversionTranslationBase):
    def __init__(self):
        translation.ConversionTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        raise NotImplementedError("extract_parameters for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, graph):
        return list(map(str, src_op.bottom))

    def extract_output_names(self, src_op, graph):
        return list(map(str, src_op.top))

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
