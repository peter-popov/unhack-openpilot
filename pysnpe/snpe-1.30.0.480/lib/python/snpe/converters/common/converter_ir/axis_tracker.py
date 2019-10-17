# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from snpe.converters.common.utils import code_to_message, snpe_converter_utils
from snpe.converters.common.utils.snpe_converter_utils import *
from snpe.converters.common.converter_ir import op_adapter


class AxisTracker(object):
    """
    This class holds all enums, modules and subclasses needed to handle axis tracking between
    source framework to our desired format.
    """
    # Class (an another way of enum class)
    # TBD: Once minimum python version is upgraded for converter from 2.7 to 3.0
    #      replace with enum class
    class AxisAnnotations(object):
        """
        This class contains axis annotations required for axis tracking.
        """
        HEIGHT = 0
        WIDTH = 1
        CHANNEL = 2
        BATCH = 3
        TIME = 4
        FEATURE = 5
        # NONTRIVIAL indicates none of axis annotation is valid and not trivial to be derived
        # Layers such as reshape/flatten specify this axis annotation.
        NONTRIVIAL = 7

    class AxisFormat(object):
        """
        Contains axis commonly used axis orders along with permute order to go to/from this well-defined formats
        """
        # Batch,Channel,Spatial. With one batch and two spatial dimensions,
        # equivalent to NCHW
        NCS = 'NCS'
        # Batch,Spatial,Channel. With one batch and two spatial dimensions,
        # equivalent to NHWC. This is the native data order for SNPE ops which
        # output feature maps.
        NSC = 'NSC'
        # Time,Batch,Feature.
        TBF = 'TBF'
        # Batch,Time,Feature. This is the native data order for SNPE RNN ops.
        BTF = 'BTF'
        # Batch,Feature.
        FEATURE = 'FEATURE'
        # Op specific data format.
        NONTRIVIAL = 'NONTRIVIAL'
        # Enum value used by buffers which have not yet undergone axis tracking.
        NOT_YET_DEFINED = 'NOT_YET_DEFINED'

        # well-known permute orders
        NCS_TO_NSC = [0, 2, 3, 1]
        NSC_TO_NCS = [0, 3, 1, 2]
        TBF_TO_BTF = BTF_TO_TBF = [1, 0, 2]

    @classmethod
    def get_axis_annotation_from_format(cls, axis_format):
        if axis_format == cls.AxisFormat.NCS:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        elif axis_format == cls.AxisFormat.NSC:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        elif axis_format == cls.AxisFormat.TBF:
            return [AxisTracker.AxisAnnotations.TIME, AxisTracker.AxisAnnotations.BATCH,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.BTF:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.TIME,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.FEATURE:
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE]
        elif axis_format == cls.AxisFormat.NONTRIVIAL:
            return [AxisTracker.AxisAnnotations.NONTRIVIAL]

        raise ValueError("Unknown axis format {}" % axis_format)

    @classmethod
    def get_axis_format_from_annotation(cls, axis_annotation):
        if axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.CHANNEL,
                               cls.AxisAnnotations.HEIGHT, cls.AxisAnnotations.WIDTH]:
            return cls.AxisFormat.NCS
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.HEIGHT,
                                 cls.AxisAnnotations.WIDTH, cls.AxisAnnotations.CHANNEL]:
            return cls.AxisFormat.NSC
        elif axis_annotation == [cls.AxisAnnotations.TIME, cls.AxisAnnotations.BATCH,
                                 cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.TBF
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.TIME,
                                 cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.BTF
        elif axis_annotation == [cls.AxisAnnotations.BATCH, cls.AxisAnnotations.FEATURE]:
            return cls.AxisFormat.FEATURE
        else:
            return cls.AxisFormat.NONTRIVIAL

    @classmethod
    def get_permute_order(cls, src_order, target_order, rank):
        if src_order == cls.AxisFormat.NCS:
            if target_order == cls.AxisFormat.NSC:
                if rank == 4:
                    return cls.AxisFormat.NCS_TO_NSC
                num_spatial = rank-2
                return [0] + [i+2 for i in range(num_spatial)] + [1]
        elif src_order == cls.AxisFormat.NSC:
            if target_order == cls.AxisFormat.NCS:
                if rank == 4:
                    return cls.AxisFormat.NSC_TO_NCS
                num_spatial = rank-2
                return [0, rank-1] + [i+1 for i in range(num_spatial)]
        elif src_order == cls.AxisFormat.TBF:
            if target_order == cls.AxisFormat.BTF:
                return cls.AxisFormat.TBF_TO_BTF
        elif src_order == cls.AxisFormat.BTF:
            if target_order == cls.AxisFormat.TBF:
                return cls.AxisFormat.BTF_TO_TBF
        else:
            raise ValueError("No permutation from %s to %s" % (src_order, target_order))

    @staticmethod
    def compute_permute_order(current_order, expected_order):
        snpe_converter_utils.log_debug("Current Axes=" + str(current_order) + " Expected Axes=" + str(expected_order))
        log_assert(set(current_order) == set(expected_order),
                   "Error: computing permute order for current and expected axes orders: values do not match;"
                   " Current order " + str(current_order) + " Expected order:" + str(expected_order) +
                   ". Make sure you are using correct Axis Annotations for orders.")
        permute_order = []
        for axis in expected_order:
            permute_order.append(current_order.index(axis))
        return permute_order

    @staticmethod
    def permute_shape(shape, order):
        return [shape[i] for i in order]

    @staticmethod
    def inject_implicit_permute(graph, input_name, target_format, permute_order, consumers=None):
        permute_name = input_name + '.' + target_format.lower()
        input_buf = graph.get_buffer(input_name)
        snpe_converter_utils.log_assert(input_buf.rank() == len(permute_order),
                                        "Error: length of buf to permute({}) does not match length of permute order({})"
                                        " for input name: {}",
                                        input_buf.rank(), len(permute_order), input_name)
        implicit_permute = op_adapter.PermuteOp(permute_name, permute_order)
        graph.inject(implicit_permute, input_name, permute_name, consumers)
        # since the implicit permute won't be visited in this pass, go
        # ahead and set the correct order for its buffer here.
        permute_buf = graph.get_buffer(permute_name)
        permute_buf.axis_format = target_format

    @classmethod
    def enforce_input_type(cls, graph, input_name, target_format, permute_order):
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == cls.AxisFormat.NONTRIVIAL:
            if input_buf.rank() == len(permute_order):
                cls.inject_implicit_permute(graph, input_name, target_format, permute_order)
            else:
                snpe_converter_utils.log_debug2("inject_implicit_permute ignored for NONTRIVIAL axis format due to rank"
                                                "({}) and permute_order({}) mismatch for input name: {}",
                                                input_buf.rank(), len(permute_order), input_name)
        elif input_buf.axis_format == cls.AxisFormat.FEATURE:
            pass
        elif input_buf.axis_format != target_format:
            raise ValueError(code_to_message.get_error_message('ERROR_INPUT_DATA_ORDER_UNEXPECTED')
                             (input_name, target_format, input_buf.axis_format))

    @classmethod
    def image_to_spatial_first_order(cls, node, graph):
        """Axis transformation for layers which take in and emit only image-valued data"""
        cls.log_axes_to_spatial_first_order(node, graph)

        # (1) if any of our inputs are NONTRIVIAL, put a permute
        # of NCS -> NSC in front of them. This will be shared
        # with everyone who consumes that buffer, so don't specify consumers
        for name in node.input_names:
            # fetch input buffers one by one to avoid degenerate case where
            # an op uses the same input more than once and needs to permute it.
            cls.enforce_input_type(graph, name, cls.AxisFormat.NSC, cls.AxisFormat.NCS_TO_NSC)

        # (2) Update all of our output buffers to be in NSC order.Output buffer is not
        # explicitly checked, it is assumed to be in NCS order.
        for buf in graph.get_output_buffers(node):
            buf.shape = cls.permute_shape(buf.shape, cls.AxisFormat.NCS_TO_NSC)
            buf.axis_format = cls.AxisFormat.NSC
            node.op.output_shape = buf.shape

    @classmethod
    def feature_to_spatial_first_order(cls, node, graph):
        # Not much to do here, just mark the outputs
        for buf in graph.get_output_buffers(node):
            buf.axis_format = cls.AxisFormat.FEATURE

    @classmethod
    def time_series_to_spatial_first_order(cls, node, graph):
        for name in node.input_names:
            cls.enforce_input_type(graph, name, cls.AxisFormat.BTF, cls.AxisFormat.TBF_TO_BTF)

        for buf in graph.get_output_buffers(node):
            if buf.rank() == 3:
                buf.shape = cls.permute_shape(buf.shape, cls.AxisFormat.TBF_TO_BTF)
                buf.axis_format = cls.AxisFormat.BTF
            elif buf.rank() == 4:
                buf.axis_format = cls.AxisFormat.NSC

    @classmethod
    def eltwise_to_spatial_first_order(cls, node, graph):
        input_buffers = graph.get_input_buffers(node)
        input_orders = [buf.axis_format for buf in input_buffers]
        if cls.AxisFormat.NSC in input_orders:
            cls.image_to_spatial_first_order(node, graph)
        elif cls.AxisFormat.BTF in input_orders:
            cls.time_series_to_spatial_first_order(node, graph)
        elif cls.AxisFormat.FEATURE in input_orders:
            cls.feature_to_spatial_first_order(node, graph)
        else:
            # well hopefully someone knows
            for buf in graph.get_output_buffers(node):
                buf.axis_format = cls.AxisFormat.NONTRIVIAL

    @staticmethod
    def log_axes_to_spatial_first_order(node, graph):
        snpe_converter_utils.log_debug(code_to_message.get_debugging_message("DEBUG_AXES_TO_SPATIAL_FIRST_ORDER_ENTRY")
                                       (node.op.name))
        for input_name in node.input_names:
            snpe_converter_utils.log_debug(
                code_to_message.get_debugging_message("DEBUG_AXES_TO_SPATIAL_FIRST_ORDER_INPUT_SIZE")
                (input_name, str(graph.get_buffer(input_name).shape)))
