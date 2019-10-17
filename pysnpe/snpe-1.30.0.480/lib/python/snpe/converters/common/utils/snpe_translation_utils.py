#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from math import ceil, floor
from .snpe_converter_utils import *
from . import code_to_message


# ---------------------------------
# Utils for calculating output dims
# ---------------------------------
def calc_conv_output_dim(input_size, filter_size, padding, stride, dilation, padding_size_strategy):

    kernel_extent = dilation * (filter_size - 1) + 1
    full_size = float(2 * padding) + input_size - kernel_extent

    if padding_size_strategy == "PADDING_SIZE_IMPLICIT_VALID":
        filter_ = int(filter_size + ((filter_size - 1) * (dilation - 1)))
        output_dim = ceil(float(input_size - filter_ + 1) / float(stride))
    elif padding_size_strategy == "PADDING_SIZE_IMPLICIT_SAME":
        output_dim = ceil(float(input_size) / float(stride))
    elif padding_size_strategy == "PADDING_SIZE_EXPLICIT_FLOOR":
        output_dim = 1 + floor(full_size/float(stride))
    else:  # EXPLICIT or UNDEFINED
        output_dim = 1 + (full_size / float(stride))

    return int(output_dim)


def calc_deconv_output_dim(input_size,
                           filter_size,
                           stride,
                           pad):
    return stride*(input_size-1) + filter_size - 2*pad  # + output_pad


def calc_pool_output_dim(input_size, pool_size, padding, stride, padding_size_strategy):
    padding = -padding
    full_size = float(input_size - 2 * padding - pool_size)

    if padding_size_strategy == "PADDING_SIZE_IMPLICIT_VALID":
        output_dim = ceil(1 + full_size) / stride
    elif padding_size_strategy == "PADDING_SIZE_IMPLICIT_SAME":
        output_dim = ceil(float(input_size) / stride)
    elif padding_size_strategy == "PADDING_SIZE_EXPLICIT_FLOOR":
        output_dim = 1 + floor(full_size/stride)
    elif padding_size_strategy == "PADDING_SIZE_EXPLICIT_ASYMMETRIC":
        # this is implemented for EXPLICIT_RIGHTHANDED in snpe c++ but modeltools maps
        # asymmetric to righthanded so mimicking that here
        full_size = float(input_size - padding - pool_size)
        output_dim = 1 + floor(full_size / stride)
    else:  # EXPLICIT or UNDEFINED
        output_dim = 1 + ceil(full_size / stride)

    if (output_dim - 1) * stride + padding >= input_size:
        # don't start a pool beyond the right border of the image
        output_dim -= 1

    return int(output_dim)


def get_conv_output_shape(ir_op, input_shapes):
    input_height = input_shapes[0][2]
    input_width = input_shapes[0][3]

    output_height = calc_conv_output_dim(input_height,
                                         ir_op.weights.shape[2],
                                         ir_op.pady,
                                         ir_op.stridey,
                                         ir_op.dilationy,
                                         ir_op.padding_size_strategy)
    output_width = calc_conv_output_dim(input_width,
                                        ir_op.weights.shape[3],
                                        ir_op.padx,
                                        ir_op.stridex,
                                        ir_op.dilationx,
                                        ir_op.padding_size_strategy)
    output_depth = ir_op.bias.shape[0]
    batch = input_shapes[0][0]
    output_shape = [batch, output_depth, output_height, output_width]
    log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(ir_op.name, output_shape))
    return [output_shape]


def get_deconv_output_shape(ir_op, input_shapes):
    input_shape = input_shapes[0]
    if ir_op.output_height == 0:
        # calculate according to provided formula
        input_height = input_shape[2]
        input_width = input_shape[3]

        output_height = calc_deconv_output_dim(input_height,
                                               ir_op.weights.shape[2],
                                               ir_op.stride,
                                               ir_op.pady)
        ir_op['output_height'] = output_height

        output_width = calc_deconv_output_dim(input_width,
                                              ir_op.weights.shape[3],
                                              ir_op.stride,
                                              ir_op.padx)
        ir_op['output_width'] = output_width
    else:
        output_height = ir_op.output_height
        output_width = ir_op.output_width

    output_depth = ir_op.bias.shape[0]
    batch = input_shapes[0][0]
    output_shape = [batch, output_depth, output_height, output_width]
    log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(ir_op.name, output_shape))
    return [output_shape]


def get_pool_output_shape(ir_op, input_shapes):
    input_shape = input_shapes[0]
    input_height = input_shape[2]
    input_width = input_shape[3]
    output_height = calc_pool_output_dim(input_height,
                                         ir_op.size_y,
                                         ir_op.pad_y,
                                         ir_op.stride_y,
                                         ir_op.padding_size_strategy)
    output_width = calc_pool_output_dim(input_width,
                                        ir_op.size_x,
                                        ir_op.pad_x,
                                        ir_op.stride_x,
                                        ir_op.padding_size_strategy)

    output_shape = input_shape[0:2] + [output_height, output_width]
    log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(ir_op.name, output_shape))
    return [output_shape]


def expand_to_rank(shape, rank):
    """
    :type shape: list[int]
    :type rank: int
    :rtype: list[int]
    """
    result = shape[:]
    while len(result) < rank:
        result.insert(0, 1)
    return result


# ------------------------------
# Util used for common squashing
# ------------------------------
def squash_nodes_into_previous(graph, matched_node_list, msg):
    """
     Squashes a nodes weights and biases (which are determined only if the previous op has weights and biases)
     arithmetically by adding the two biases or multiplying the weights. Intended use is for elementwise ops
     that follow batchnorm, FC or convolution.
    :param graph: The IROpGraph object
    :param matched_node_list: the list of nodes that are elementwise ops, have a constant input, and are preceded by a
                              batchnorm ,FC or convolution.
    :param msg: The debug message to be printed.

    """

    for node_tuple in matched_node_list:

        # collect previous and current op information
        node = node_tuple[0]
        input_buffer = graph.get_input_buffers(node)[0]
        prev = input_buffer.producer

        # we need separate conditionals since the arithmetic is different for addition/subtraction
        # vs multiplication/division
        if node.op.type == 'elementwise_sum' or node.op.type == 'elementwise_sub':
            scale_bias = node.op.bias
            prev.op.bias += scale_bias
        elif node.op.type == 'elementwise_div' or node.op.type == 'elementwise_product':
            scale_weights = node.op.weights
            prev.op.weights *= scale_weights
            prev.op.bias = (prev.op.bias * scale_weights)
        elif node.op.type == "scale":
            scale_weights = node.op.weights
            scale_bias = node.op.bias
            prev.op.weights *= scale_weights
            prev.op.bias = (prev.op.bias * scale_weights) + scale_bias
        else:
            continue

        graph.squash(node, input_buffer.name)
        log_debug2(code_to_message.get_debugging_message(msg)(node.op.name,
                                                              prev.op.name,
                                                              prev.op.type))


# -------------------------------------------------------
# Util used for mapping framework activation to snpe enum
# -------------------------------------------------------
def extract_activation(activation):
    acts = {'RELU': "NEURON_RELU",
            'TANH': "NEURON_TANH",
            'SIGMOID': "NEURON_LOGISTIC",
            'ELU': "NEURON_ELU"}
    try:
        return acts[str(activation).upper()]
    except KeyError:
        raise ValueError(code_to_message.get_error_message("ERROR_ACTIVATION_FUNCTION_UNSUPPORTED")(activation))
