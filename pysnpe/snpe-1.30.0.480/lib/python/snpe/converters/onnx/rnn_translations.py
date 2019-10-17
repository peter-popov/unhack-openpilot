# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from .onnx_translations import *
from snpe.converters.common.utils import snpe_translation_utils


# ------------------------------------------------------------------------------
#  RNNTranslationBase
# ------------------------------------------------------------------------------
OPTIONAL_INPUTS = NamedDict(initial_c='', initial_h='')
RNN_OUTPUT_TYPES = ('_all_hidden', '_final_hidden', '_final_cell')


class OnnxRnnTranslationsBase(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []
        self.weights = []
        self.rec_weights = []
        self.bias = None
        self.params = NamedDict()
        self.no_of_gates = 1
        self.backward = False
        self.output_names = []

    def extract_params_for_type(self, src_op, graph):
        self.input_names = list(map(str, src_op.input))
        self.output_names = self.extract_output_names(src_op, graph)
        self.params.direction = str(self.params.direction).lower()
        self.weights, self.rec_weights = graph.weights.fetch(*self.input_names[1:3])
        self.backward = False if self.params.direction is 'forward' else True

        # all inputs are required for caffe2 backend,  but we do not need
        # nor use sequence length, deleting from graph. Cleanup to check
        # the parameters we do support
        graph.weights.weight_map.pop('seq')

        if len(self.input_names) >= 4:
            self.bias = graph.weights.fetch(self.input_names[3])

        # If its bi-directional, we include support for custom activations (although
        # not available in snpe as yet). Also check that weights and rec_weights
        # have the right shape
        if self.params.direction == "bidirectional":
            self.params.activations = list(map(snpe_translation_utils.extract_activation,
                                          self.params.activations))
            log_assert(self.weights.shape[0] == 2 and self.rec_weights.shape[0] == 2,
                       "Node {}: Bidirectional input requires two sets of weights and recurrent weights each. "
                       "Got only {} set of weights",
                       src_op.name, self.weights.shape[0])
        else:
            # Limit the length of the user defined activations to the number of gates if unidirectional
            self.params.activations = list(map(snpe_translation_utils.extract_activation,
                                          self.params.activations[0:self.no_of_gates]))

    def convert_params_to_snpe(self, weights, rec_weights, bias, hidden_size):

        no_of_gates = self.no_of_gates

        if bias is None:
            bias = np.zeros((no_of_gates, hidden_size), dtype=numpy.float32)
        else:
            # for probably vendor specific reasons, ONNX defines GRU bias to
            # be separated into forward and recurrent parts, that are always
            # added together (unless linear_before_reset is false, but we
            # don't support that). So we will always combine.
            # We need to reshape bias which is in (2*no_of_gates*hidden_size)
            # into (hidden_size, 2*no_of_gates).
            bias = np.reshape(bias, (hidden_size, 2 * no_of_gates))
            new_bias = np.empty((no_of_gates, hidden_size), dtype=numpy.float32)
            # Elements are stored in [weights, rec_weights] where each column
            # represents the gate and the number of rows is the hidden size
            for i in range(no_of_gates):
                np.add(bias[:, i], bias[:, i + no_of_gates], out=new_bias[i, :])
            bias = new_bias

        # weights and rec_weights are also laid out as (no_of_gates*hidden_size, input_size)
        # and (no_of_gates*hidden_size, hidden_size)respectively. We need to reshape
        # to SNPE format depending on the rnn type.
        weights = np.reshape(weights, (no_of_gates, hidden_size, weights.shape[-1]), 'F')
        rec_weights = np.reshape(rec_weights, (no_of_gates, hidden_size, hidden_size), 'F')

        return weights, rec_weights, bias

    def extract_input_names(self, src_op, graph):

        # empty string for initial_h (initial hidden state) and initial_c (initial cell state)
        # means we don't need to add the input buffer to the ir_graph.
        formatted_input_names = [self.input_names[0], OPTIONAL_INPUTS.initial_h, OPTIONAL_INPUTS.initial_c]

        return [name for name in formatted_input_names if name is not '']

    def extract_output_names(self, src_op, graph):
        return [output for i, output in enumerate(src_op.output) if output]

    def infer_output_shapes(self, op, input_shapes):

        # There is a different output shape for each buffer depending on the direction, as well as
        # the number of outputs.
        # 1 output name -> Y : hidden state for all time steps
        # 2 output names - > Y, Y_h : Y same as above while Y_h is the final hidden state
        # 3 output names -> Y, Y_h, Y_c: Y_c is the final cell state

        y_unidirectional_shapes = [[input_shapes[0][0], 1, input_shapes[0][1], self.params.hidden_size],
                                    [1, input_shapes[0][1], self.params.hidden_size],
                                    [1, input_shapes[0][1], self.params.hidden_size]]

        output_shapes = [y_unidirectional_shapes[i] for i, _ in enumerate(self.output_names)]

        return output_shapes

    def create_rnn(self, src_op, graph, create_unidirectional_func, create_bidirectional_func):

        if self.params.direction == "bidirectional":
            create_bidirectional_func(src_op, graph)
        else:
            op = create_unidirectional_func()
            input_names = self.extract_input_names(src_op, graph)
            output_names = self.extract_output_names(src_op, graph)
            node = graph.add(op, input_names, output_names)
            self.populate_axes_format(node, graph)

    def create_bidirectional_module(self, src_op, graph, weights, rec_weights, bias, params, create_rnn_type):
        # set up forward op
        forward_op = create_rnn_type(src_op.op_type + '_forward',
                                     weights=weights[0, :, :],
                                     rec_weights=rec_weights[0, :, :],
                                     bias=bias[0, :] if bias.all() else None,
                                     hidden_size=params.hidden_size,
                                     backward=False)

        # set up backward op
        backward_op = create_rnn_type(src_op.op_type + '_backward',
                                      weights=weights[1, :, :],
                                      rec_weights=rec_weights[1, :, :],
                                      bias=bias[1, :] if bias.all() else None,
                                      hidden_size=params.hidden_size,
                                      backward=True)
        # set up concat ops
        concat_ops = [op_adapter.ConcatOp(src_op.op_type + '_concat_1', axis=3),
                      op_adapter.ConcatOp(src_op.op_type + '_concat_2', axis=0),
                      op_adapter.ConcatOp(src_op.op_type + '_concat_3', axis=0)]

        # set up naming so that the buffers are all different and tagged correctly
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        backward_output_names = [str(name) + RNN_OUTPUT_TYPES[i] + "_backward" for i, name in enumerate(output_names)]
        forward_output_names = [str(name) + RNN_OUTPUT_TYPES[i] + "_forward" for i, name in enumerate(output_names)]
        concat_output_names = [str(name) for name in output_names]

        # Insert the backward op, should be connected with the previous node
        backward_node = graph.add(backward_op, input_names, backward_output_names)
        self.populate_axes_format(backward_node, graph)

        # assign the forward op to the previous node
        forward_node = graph.add(forward_op, input_names, forward_output_names)
        self.populate_axes_format(forward_node, graph)

        # add concat op to the graph, should end up as a child of both forward and backward ops.
        # we need more than one concat node, depending on the number of outputs.
        for i in range(0, len(output_names)):
            concat_node = graph.add(concat_ops[i], [forward_output_names[i], backward_output_names[i]],
                                     [concat_output_names[i]])
            self.populate_axes_format(concat_node, graph)


# ------------------------------------------------------------------------------
#  GRU
# ------------------------------------------------------------------------------

class OnnxGruTranslation(OnnxRnnTranslationsBase):
    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        self.register_op_schema('GRU', [1, 7], [['clip', 'activation_alpha', 'activation_beta', 'output_sequence']])
        self._op_schema.replace_default_values(activations=['Sigmoid', 'Sigmoid', 'Tanh']*2)
        self._op_schema.register_method(self.validate_attribute_values)
        self.no_of_gates = 3

    def extract_parameters(self, src_op, graph):
        self.params = extract_attributes(src_op, schema=self.op_schema())

        self.extract_params_for_type(src_op, graph)

        # gru output is only for Y
        log_assert(self.output_names == 1,
                   "Node {}: SNPE only supports the first output, Y, of the Onnx GRU Op",
                   src_op.name)

        if len(self.input_names) >= 6:
            # check if initial_h is included
            OPTIONAL_INPUTS.initial_h = self.input_names[5]

    def add_op(self, src_op, graph):
        self.extract_parameters(src_op, graph)
        self.create_rnn(src_op, graph, self.create_unidirectional_gru,
                        self.create_bidirectional_gru)

    def create_unidirectional_gru(self, name='gru', **kargs):

        if kargs:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(kargs['weights'], kargs['rec_weights'],
                                                                       kargs['bias'], kargs['hidden_size'])
        else:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(self.weights,
                                                                       self.rec_weights,
                                                                       self.bias,
                                                                       self.params.hidden_size)
        # gru specific organization into separate gates
        activations = self.params.activations
        control_gate = {'weights': weights[0, :, :].T,
                        'rec_weights': rec_weights[0, :, :].T,
                        'bias': bias[0, :]}
        forget_gate = {'weights': weights[1, :, :].T,
                       'rec_weights': rec_weights[1, :, :].T,
                       'bias': bias[1, :]}
        state_gate = {'weights': weights[2, :, :].T,
                      'rec_weights': rec_weights[2, :, :].T,
                      'bias': bias[2, :]}

        return op_adapter.GruOp(name,
                                state_gate,
                                forget_gate,
                                control_gate,
                                activation=activations[0],
                                gate_activation=activations[1],
                                rec_gate_activation=activations[2],
                                h_0_input_name=OPTIONAL_INPUTS.initial_h,
                                backward=self.backward if not kargs else kargs['backward'])

    def create_bidirectional_gru(self, src_op, graph):
        return self.create_bidirectional_module(src_op, graph, self.weights, self.rec_weights, self.bias,
                                                self.params, self.create_unidirectional_gru)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'activations' or attr_name == 'linear_before_reset':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)


OnnxTranslations.register_translation(OnnxGruTranslation(),
                                      converter_type('GRU', 'onnx'),
                                      op_adapter.GruOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#  LSTM
# ------------------------------------------------------------------------------
class OnnxLSTMTranslation(OnnxRnnTranslationsBase):
    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        self.register_op_schema('LSTM', [1, 7], [['clip', 'activation_alpha', 'activation_beta', 'output_sequence']])
        self._op_schema.replace_default_values(activations=['Sigmoid', 'Sigmoid', 'Sigmoid', 'Tanh'] * 2)
        self._op_schema.register_method(self.validate_attribute_values)
        self.no_of_gates = 4
        self.peephole_weights = []

    def extract_parameters(self, src_op, graph):
        self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        # set parameters
        self.extract_params_for_type(src_op, graph)

        if len(self.input_names) >= 7 and len(self.output_names) == 3:
            # check if initial_h and initial_c are included
            # snpe requires that if they are included, then all
            # 3 outputs will be returned.
            OPTIONAL_INPUTS.initial_c = self.input_names[6]
            OPTIONAL_INPUTS.initial_h = self.input_names[5]

    def add_op(self, src_op, graph):
        self.extract_parameters(src_op, graph)
        self.create_rnn(src_op, graph, self.create_unidirectional_lstm,
                        self.create_bidirectional_lstm)

    def create_unidirectional_lstm(self, name='lstm', **kargs):

        if kargs:
            [gate_weights, gate_rec_weights, gate_bias] = self.convert_params_to_snpe(kargs['weights'],
                                                                                      kargs['rec_weights'],
                                                                                      kargs['bias'],
                                                                                      kargs['hidden_size'])
        else:
            [gate_weights, gate_rec_weights, gate_bias] = self.convert_params_to_snpe(self.weights,
                                                                                      self.rec_weights,
                                                                                      self.bias,
                                                                                      self.params.hidden_size)
        # LSTM specific organization into gate format
        gate_bias = gate_bias.reshape(-1,)
        gate_rec_weights = gate_rec_weights.reshape(self.no_of_gates*self.params.hidden_size, -1)
        gate_weights = gate_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)

        return op_adapter.LstmOp(name,
                                 gate_weights=gate_weights,
                                 recurrent_weights=gate_rec_weights,
                                 gate_bias=gate_bias,
                                 backward=self.backward if not kargs else kargs['backward'],
                                 c_0_input_name=OPTIONAL_INPUTS.initial_c,
                                 h_0_input_name=OPTIONAL_INPUTS.initial_h,
                                 reset_state_at_time_step_0=True)

    def create_bidirectional_lstm(self, src_op, graph):
            return self.create_bidirectional_module(src_op, graph, self.weights, self.rec_weights, self.bias,
                                                    self.params, self.create_unidirectional_lstm)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'activations' or attr_name == 'input_forget':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)


OnnxTranslations.register_translation(OnnxLSTMTranslation(),
                                      converter_type('LSTM', 'onnx'),
                                      op_adapter.LstmOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
# RNN
# ------------------------------------------------------------------------------

class OnnxRNNTranslation(OnnxRnnTranslationsBase):
        def __init__(self):
            OnnxRnnTranslationsBase.__init__(self)
            self.register_op_schema('RNN', [1, 7], [['clip', 'activation_alpha', 'activation_beta']])
            self._op_schema.register_method(self.validate_attribute_values)
            self.no_of_gates = 1

        def extract_parameters(self, src_op, graph):
            self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

            self.extract_params_for_type(src_op, graph)

        def add_op(self, src_op, graph):
            self.extract_parameters(src_op, graph)
            self.create_rnn(src_op, graph, self.create_unidirectional_rnn,
                            self.create_bidirectional_rnn)

        def create_unidirectional_rnn(self, name='rnn', **kargs):

            if kargs:
                [weights, _, bias] = self.convert_params_to_snpe(kargs['weights'],
                                                                 kargs['rec_weights'],
                                                                 kargs['bias'],
                                                                 kargs['hidden_size'])
            else:
                [weights, _, bias] = self.convert_params_to_snpe(self.weights,
                                                                 self.rec_weights,
                                                                 self.bias,
                                                                 self.params.hidden_size)
            return op_adapter.RnnTransformationOp(name,
                                                  weights=weights,
                                                  bias=bias,
                                                  activation=self.params.activations)

        def create_bidirectional_rnn(self, src_op, graph):
            return self.create_bidirectional_module(src_op, graph, self.weights, self.rec_weights, self.bias,
                                                    self.params, self.create_unidirectional_rnn)

        @staticmethod
        def validate_attribute_values(src_op, attr_name, attr_value):
            if attr_name == 'activations':
                OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)


OnnxTranslations.register_translation(OnnxRNNTranslation(),
                                      converter_type('RNN', 'onnx'),
                                      op_adapter.RnnTransformationOp.TRANSLATION_KEY)
