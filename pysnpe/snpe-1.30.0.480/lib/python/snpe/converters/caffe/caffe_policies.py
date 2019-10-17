# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from snpe.converters.common.converter_ir import op_policies, translation
from snpe.converters.common.utils import code_to_message
from snpe.converters.common.utils.snpe_converter_utils import *
from .caffe_base_translation import CaffeTranslations


# ------------------------------------------------------------------------------
#   Policies
# ------------------------------------------------------------------------------
class CaffeNamePolicy(op_policies.ConversionNamePolicy):
    """
    Keep track of all ops and fix op names and inputs/outputs names such that if more than one op output
    has the same name, subsequent outputs and the following inputs with the same name get renamed. This is done
    because in-place is not supported.
    Eg:
    layer {
      name: "bn1"
      type: "batchnorm"
      bottom: "data"
      top: "bn1"
    }
    layer {
      name: "bn_scale1"
      type: "scale"
      bottom: "bn1"
      top: "bn1"     <-- get_output_names will return this as bn_scale1.bn1
    }
    layer {
      name: "output"
      type: "relu"
      bottom: "bn1"  <-- get_input_names will return this as bn_scale1.bn1
      top: "relu1"
    }
    """

    def __init__(self):
        op_policies.ConversionNamePolicy.__init__(self)
        self.output_map = {}

    def get_op_name(self, op):
        if op.name:
            return str(op.name)
        else:
            count = self.type_count.get(op.type, 0)
            self.type_count[op.type] = count+1
            return "%s_%d" % (op.type, count)

    def get_input_names(self, op, input_names):
        _input_names = input_names[:]  # deep copy as it can change layer.bottom
        for index, input_name in enumerate(_input_names):
            # If a mapping for an output exists and it's different remap the input to the new name
            if input_name in self.output_map and input_name != self.output_map[input_name]:
                log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERT_REMAP_INPUT')
                          (op.name, input_name, self.output_map[input_name]))
                _input_names[index] = self.output_map[input_name]
        return list(map(str, _input_names))

    def get_output_names(self, op, output_names):
        # Process outputs next, they may remap later inputs
        _output_names = output_names[:]  # deep copy as it can change layer.top
        for index, output_name in enumerate(_output_names):
            if output_name not in self.output_map:
                self.output_map[output_name] = output_name
            else:
                # op name will work to make output_name unique because Caffe enforces unique layer names
                output_name_alias = str(op.name) + "." + str(output_name)
                log_debug(code_to_message.get_debugging_message('DEBUG_CAFFE_CONVERT_REMAP_OUTPUT')
                          (op.name, output_name, output_name_alias))
                self.output_map[output_name] = output_name_alias
                _output_names[index] = output_name_alias
        return list(map(str, _output_names))

    def get_caffe_name_mapping(self, name):
        log_assert(name in self.output_map, "Input or Output name {} not found in registered graph.", name)
        return self.output_map[name]


class CaffeShapeInferencePolicy(op_policies.ConversionShapeInferencePolicy):

    def infer_shape(self, op, input_shapes):
        return CaffeTranslations.apply_method_to_op(op.type, translation.INFER_SHAPE, op, input_shapes)