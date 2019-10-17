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


# -----------------------------------------------------------------
# Converter translations
# -----------------------------------------------------------------
class CaffeLstmTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        input_names = graph.naming_policy.get_input_names(layer, layer.bottom)
        sequence_continuation_name = ''
        if len(input_names) > 1:
            sequence_continuation_name = input_names[1]

        x_static_name = ''
        if len(input_names) in (3, 5):
            x_static_name = input_names[2]

        c_0_input_name = ''
        h_0_input_name = ''
        if len(input_names) > 3:
            c_0_input_name = input_names[-2]
            h_0_input_name = input_names[-1]

        x_weights, bias, h_weights = graph.weights.get_lstm_weights(layer)
        return op_adapter.LstmOp(layer.name,
                                 gate_weights=x_weights,
                                 gate_bias=bias,
                                 recurrent_weights=h_weights,
                                 w_xc_static=0,
                                 backward=False,
                                 reset_state_at_time_step_0=False,
                                 h_0_input_name=h_0_input_name,
                                 c_0_input_name=c_0_input_name,
                                 sequence_continuation_name=sequence_continuation_name,
                                 x_static_name=x_static_name
                                 )

    def extract_output_names(self, layer, graph):
        name = str(layer.name)
        input_names = graph.naming_policy.get_input_names(layer, layer.bottom)
        output_names = [name]
        if len(input_names) > 3:
            output_names.append('{}_c_T'.format(name))
            output_names.append('{}_h_T'.format(name))

        return output_names

    def infer_output_shapes(self, op, input_shapes):
        time_steps = 1
        streams = 1
        output_dims = []
        # TBF for shape since axes_to_spatial_first not done yet
        if len(input_shapes[0]) == 3:
            time_steps = input_shapes[0][0]
            streams = input_shapes[0][1]
        output_depth = op.recurrent_weights.shape[1]  # this gets us recurrent_param.num_output
        output_dims.append([time_steps, streams, output_depth])

        if op.c_0_input_name and op.h_0_input_name:
            # Layer has exposed recurrent inputs, therefore we need to add c_T and h_T outputs
            output_dims.append([streams, output_depth])
            output_dims.append([streams, output_depth])

        return output_dims


CaffeTranslations.register_translation(CaffeLstmTranslation(),
                                       converter_type('lstm', 'caffe'),
                                       op_adapter.LstmOp.TRANSLATION_KEY)
