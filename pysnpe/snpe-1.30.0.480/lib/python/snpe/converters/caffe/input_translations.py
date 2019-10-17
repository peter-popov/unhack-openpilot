# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from snpe.converters.common.converter_ir import op_adapter
from snpe.converters.common.utils.snpe_converter_utils import *

ADD_INPUT_OP = "ADD_INPUT_OP"


# -----------------------------------------------------------------
# Converter translations
# -----------------------------------------------------------------
class CaffeInputTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.register_method(ADD_INPUT_OP, self.add_input_op)

    def add_input_op(self, input_name, input_dims, graph):
        node = graph.add_input(input_name, input_dims)
        self.populate_axes_format(node, graph)


CaffeTranslations.register_translation(CaffeInputTranslation(),
                                       converter_type('input', 'caffe'),
                                       converter_type('data', 'caffe'),
                                       converter_type('dummydata', 'caffe'),
                                       op_adapter.InputOp.TRANSLATION_KEY)
