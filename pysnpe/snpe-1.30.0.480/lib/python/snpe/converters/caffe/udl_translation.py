# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from snpe.converters.common.converter_ir import op_adapter
from snpe.converters.common.converter_ir.axis_tracker import AxisTracker
from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from snpe.converters.common.utils import code_to_message
from snpe.converters.common.utils.snpe_converter_utils import *


class UdlTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, udl_layer, graph):
        layer = udl_layer[0]
        udl_obj = udl_layer[1]
        expected_input_axis_orders, expected_output_axis_orders = udl_obj.get_expected_axis_order()
        # set default axis orders if not given (Setting to spatial first order)
        if not len(expected_input_axis_orders):
            expected_input_axis_orders = [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                                          AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        if not len(expected_output_axis_orders):
            expected_output_axis_orders = [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                                           AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        input_names = graph.naming_policy.get_input_names(layer, layer.bottom)
        input_shapes = []

        # verify input/output ranks are supported and append shapes
        for name in input_names:
            if name not in graph.buffers:
                raise KeyError("Graph has no buffer %s, referred to as input for %s" % (name, layer.name))
            input_order_ranks = [len(dims) for dims in expected_input_axis_orders]
            log_assert(graph.buffers[name].rank() in input_order_ranks,
                       code_to_message.get_error_message('DEBUG_CAFFE_UNSUPPORTED_INPUT_DIMS')
                       (graph.buffers[name].rank(), layer.name))
            input_shapes.append(graph.buffers[name].shape)

        udl_func = udl_obj.get_layer_callback()
        blob_output = udl_func(layer, graph.weights, input_shapes)
        blob = blob_output.get_blob()

        # we need a list of lists.
        # i.e. a list of dimensions. each dimensions is a list
        output_dims = []
        for idx in range(len(layer.top)):
            # FIXME do we need list() here?
            log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_OUTPUT_DIMS_IDX')(str(idx)))
            dim = blob_output.get_output_dims(idx)
            assert(isinstance(dim, list))
            output_order_ranks = [len(dims) for dims in expected_output_axis_orders]
            log_assert(len(dim) in output_order_ranks,
                       code_to_message.get_error_message('DEBUG_CAFFE_UNSUPPORTED_OUTPUT_DIMS')
                       (len(dim), layer.name))
            output_dims.append(dim)

        log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_UDL_OUTPUT_DIMS')(str(output_dims)))
        log_assert(blob.get_size() != 0,
                   code_to_message.get_error_message('ERROR_CAFFE_UDL_BLOB_SIZE_IS_ZERO')(layer.name))

        return op_adapter.UdlOp(layer.name,
                                layer_type=str(layer.type).upper(),
                                blob=blob.get_blob(),
                                output_dims=output_dims,
                                expected_input_axis_orders=expected_input_axis_orders,
                                expected_output_axis_orders=expected_output_axis_orders)

    def extract_input_names(self, udl_layer, graph):
        layer = udl_layer[0]
        return list(map(str, layer.bottom))

    def extract_output_names(self, udl_layer, graph):
        layer = udl_layer[0]
        return list(map(str, layer.top))

    def infer_output_shapes(self, op, input_shapes):
        return op.output_dims


CaffeTranslations.register_translation(UdlTranslation(),
                                       op_adapter.UdlOp.TRANSLATION_KEY)
