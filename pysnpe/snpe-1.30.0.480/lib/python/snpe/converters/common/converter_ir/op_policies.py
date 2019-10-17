# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class ConversionNamePolicy(object):
    def __init__(self):
        self.type_count = {}

    def get_op_name(self, op):
        raise NotImplementedError("get_op_name for {} not implemented ".format(str(self.__class__.__name__)))

    def get_input_names(self, op, input_names):
        return list(map(str, input_names))

    def get_output_names(self, op, output_names):
        return list(map(str, output_names))


class ConversionShapeInferencePolicy(object):

    def infer_shape(self, op, input_shapes):
        raise NotImplementedError("infer_shape for {} not implemented ".format(str(self.__class__.__name__)))
