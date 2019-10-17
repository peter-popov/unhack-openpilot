# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import caffe

from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from snpe.converters.common.converter_ir import op_adapter
from snpe.converters.common.utils import code_to_message
from snpe.converters.common.utils.snpe_converter_utils import *

from functools import reduce

class CaffeChannelShuffleTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        groups = 1
        if hasattr(layer, "shuffle_channel_param"):
            shuffle_channel_param = layer.shuffle_channel_param
            if hasattr(shuffle_channel_param, "group"):
                groups = shuffle_channel_param.group
            else:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE_CHANNEL_SHUFFLE_LAYER_MISSING_GROUPS_ARG')
                    (str(layer.type)))
        return op_adapter.ChannelShuffleOp(layer.name,
                                           groups=groups)


CaffeTranslations.register_translation(CaffeChannelShuffleTranslation(),
                                       converter_type('shufflechannel', 'caffe'),
                                       op_adapter.ChannelShuffleOp.TRANSLATION_KEY)


class CaffeConcatTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        caffe_axis = layer.concat_param.axis

        return op_adapter.ConcatOp(layer.name,
                                   axis=caffe_axis)

    def infer_output_shapes(self, op, input_shapes):
        # Add batch dim
        axis = op.axis
        output_shape = input_shapes[0][:]
        output_shape[axis] = sum(shape[axis] for shape in input_shapes)
        return [output_shape]


CaffeTranslations.register_translation(CaffeConcatTranslation(),
                                       converter_type('concat', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.CONCAT, 'caffe'),
                                       op_adapter.ConcatOp.TRANSLATION_KEY)


class CaffeCropTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        input_name, shape_name = graph.naming_policy.get_input_names(layer, layer.bottom)
        input_dims = graph.get_buffer(input_name).get_buf_dims()
        target_dims = graph.get_buffer(shape_name).get_buf_dims()[:]  # deep copy so that input dims don't change

        caffe_offset = [int(o) for o in layer.crop_param.offset]
        if len(caffe_offset) == 0:
            caffe_offset = [0]*4
        elif len(caffe_offset) == 1:
            caffe_offset = [caffe_offset[0]]*4

        # Note: the offsets are populated based on caffe axis ordering and will later be permuted(as needed) in
        #       axis_to_spatial_first_order in optimizations
        if len(target_dims) == 4:
            axis = layer.crop_param.axis % 4
            if axis == 0:
                offsets = [caffe_offset[0], caffe_offset[1], caffe_offset[2], caffe_offset[3]]
            elif axis == 1:
                offsets = [0, caffe_offset[0], caffe_offset[1], caffe_offset[2]]
            elif axis == 2:
                offsets = [0, 0, caffe_offset[0], caffe_offset[1]]
                target_dims[0] = input_dims[0]
                target_dims[1] = input_dims[1]
            else:
                offsets = [0, 0, 0, caffe_offset[0]]
                target_dims[0] = input_dims[0]
                target_dims[1] = input_dims[1]
                target_dims[2] = input_dims[2]
        elif len(target_dims) == 3:
            axis = layer.crop_param.axis % 4
            if axis == 0:
                offsets = [caffe_offset[1], caffe_offset[2], caffe_offset[3]]
            elif axis == 1:
                offsets = [caffe_offset[0], caffe_offset[1], caffe_offset[2]]
            elif axis == 2:
                offsets = [0, caffe_offset[0], caffe_offset[1]]
                target_dims[0] = input_dims[0]
            else:
                offsets = [0, 0, caffe_offset[0]]
                target_dims[0] = input_dims[0]
                target_dims[1] = input_dims[1]
        elif len(target_dims) == 1:
            offsets = [caffe_offset[0]]
        else:
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_CROP_LAYER_OUTPUT_DIM_ERR')
                             (str(layer.name)))

        return op_adapter.CropOp(layer.name,
                                 offsets=offsets,
                                 output_shape=target_dims)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


CaffeTranslations.register_translation(CaffeCropTranslation(),
                                       converter_type('crop', 'caffe'),
                                       op_adapter.CropOp.TRANSLATION_KEY)


class CaffePermuteTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        if hasattr(layer, 'ssd_permute_param'):
            permute_param = layer.ssd_permute_param
        else:
            permute_param = layer.permute_param
        if not len(permute_param.order):
            raise ValueError(
                code_to_message.get_error_message('ERROR_CAFFE_PERMUTE_LAYER_MISSING_ORDER_FIELD')(str(layer.name)))
        permute_order = permute_param.order

        return op_adapter.PermuteOp(layer.name,
                                    order=permute_order)


CaffeTranslations.register_translation(CaffePermuteTranslation(),
                                       converter_type('permute', 'caffe'),
                                       op_adapter.PermuteOp.TRANSLATION_KEY)


class CaffeReshapeTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        # There are 2 different layers in Caffe that are mapped to the SNPE Reshape layer.
        #  - For "Reshape", the "shape" BlobShape parameter defines the output dimensions, with a 0
        #    indicating an unchanged dimension (to be copied from the corresponding input dimension,
        #    and -1 indicating all remaining dimensionality to be folded into this dimension.
        #    Additionally, Reshape has the axis parameter which specifies the first dimension to be
        #    included in the reshape operation (default 0) and the num_axis parameter which specifies
        #    how many of the dimensions to include (default -1 meaning all)
        #  - For "Flatten", the axis (default 1) and end_axis (default -1 meaning last) are used to
        #    determine which dimensions are to be folded into the single output dimension.

        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dims = graph.get_buffer(input_name).get_buf_dims()
        layer_type = converter_type(layer.type, "caffe")
        output_dims = []

        if layer_type == converter_type('reshape', 'caffe'):
            input_size = reduce(int.__mul__, input_dims)
            output_dims = list(map(int, layer.reshape_param.shape.dim))
            axis = layer.reshape_param.axis
            num_axes = layer.reshape_param.num_axes
            if axis < 0:
                axis = len(input_dims) + axis
            if num_axes < 0:
                num_axes = len(input_dims) - axis

            # replace any 0 in the output_dims with the corresponding dimension in the input_dims.
            output_dims = list(map(lambda x: input_dims[x + axis] if output_dims[x] == 0 else output_dims[x],
                                   range(len(output_dims))))
            # prefix/postfix
            output_dims = input_dims[:axis] + output_dims + input_dims[axis+num_axes:]

            # replace -1 in the output by the remainder of the inputs
            remainder_index = [i for i, j in enumerate(output_dims) if j == -1]
            if len(remainder_index) == 1:
                output_size = -1 * reduce(int.__mul__, output_dims)  # multiply by -1 to make this positive
                output_dims[remainder_index[0]] = input_size / output_size

        if layer_type == converter_type('flatten', 'caffe'):
            axis = layer.flatten_param.axis
            end_axis = layer.flatten_param.end_axis
            if axis < 0:
                axis = len(input_dims) + axis
            if end_axis < 0:
                end_axis = len(input_dims) + end_axis
            output_dims = [reduce(int.__mul__, input_dims[axis:end_axis+1])]
            output_dims = input_dims[:axis] + output_dims + input_dims[end_axis+1:]

        return op_adapter.ReshapeOp(layer.name,
                                    output_shape=output_dims)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


CaffeTranslations.register_translation(CaffeReshapeTranslation(),
                                       converter_type('reshape', 'caffe'),
                                       converter_type('flatten', 'caffe'),
                                       op_adapter.ReshapeOp.TRANSLATION_KEY)


class CaffeSliceTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.output_names = []

    def extract_parameters(self, layer, graph):
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dim = graph.get_buffer(input_name).get_buf_dims()
        self.output_names = layer.top  # for layer output shape inference

        # By default, slice_axis is 1
        slice_axis = 1
        if layer.slice_param.HasField('slice_dim'):
            slice_axis = layer.slice_param.slice_dim
            log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_SLICE_DIM'))
        else:
            try:
                slice_axis = layer.slice_param.axis
                log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_AXIS'))
                # Since axis parameter could contain -ve value, let's turn it to +ve
                if slice_axis < 0:
                    slice_axis = len(input_dim) + slice_axis
            except AttributeError:
                log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_DEFINE_SLICE_DIM_AXIS_FIELD'))
                log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_AXIS_DEFAULT_FOR_LAYER')
                          (str(layer.type), layer.name))
                pass

        slice_points = [int(v) for v in layer.slice_param.slice_point]

        return op_adapter.SliceOp(layer.name,
                                  axis=slice_axis,
                                  slice_points=slice_points)

    def infer_output_shapes(self, op, input_shapes):
        input_shape = input_shapes[0]
        output_dims = []

        log_assert(len(self.output_names) >= 2, "Layer {} needs >= 2 output buffers.", op.name)
        log_assert(op.axis < len(input_shape), "Layer {} slice axis {} falls outside the input dim size: {}.",
                   op.name, op.axis, len(input_shape))

        # If slice points are not specified, evenly divide the input for as many
        # requested output buffers
        slice_points = op.slice_points
        if not len(slice_points):
            slice_point_size = input_shape[op.axis]/len(self.output_names)

            next_slice_point = slice_point_size
            for output_idx in range(len(self.output_names) - 1):
                # Capture slice point
                slice_points.append(next_slice_point)
                next_slice_point += slice_point_size

        # Compute output dim for each output buffer taking into account
        # slice points and slice axis
        for output_idx in range(len(self.output_names)):
            extent = input_shape[:]  # deep copy since we will be changing extent

            if output_idx < (len(self.output_names) - 1):

                log_assert(slice_points[output_idx] < int(input_shape[op.axis]),
                           "Layer {}  slice point {} falls out of input dim {} ",
                           op.name, slice_points[output_idx], int(input_shape[op.axis]))
                if output_idx == 0:
                    extent[op.axis] = slice_points[output_idx]
                else:
                    extent[op.axis] = slice_points[output_idx] - slice_points[output_idx - 1]
            else:
                extent[op.axis] = input_shape[op.axis] - slice_points[output_idx - 1]

            output_dims.extend([extent])

        return output_dims


CaffeTranslations.register_translation(CaffeSliceTranslation(),
                                       converter_type('slice', 'caffe'),
                                       op_adapter.SliceOp.TRANSLATION_KEY)


class CaffeTileTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        caffe_axis = layer.tile_param.axis
        return op_adapter.ConcatOp(layer.name,
                                   axis=caffe_axis)

    def extract_input_names(self, layer, graph):
        caffe_tiles = layer.tile_param.tiles
        target_inames = graph.naming_policy.get_input_names(layer, layer.bottom)
        tile_inputs = target_inames * caffe_tiles

        return tile_inputs


CaffeTranslations.register_translation(CaffeTileTranslation(),
                                       converter_type('tile', 'caffe'))
