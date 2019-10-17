#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from snpe.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from snpe.converters.tensorflow.util import ConverterError
from snpe.converters.tensorflow.sequences.lstm import (
    cell_sequence,
    state_sequence
)


class LstmLayerResolver(LayerResolver, object):

    class StateDescriptor(LayerDescriptor):
        def __init__(self, name, operations, expand_dims_op_1, expand_dims_op_2):
            super(LstmLayerResolver.StateDescriptor, self).__init__('LSTM_STATE', name, operations)

            self.expand_dims_op_1 = expand_dims_op_1
            self.expand_dims_op_2 = expand_dims_op_2

        def is_input_tensor(self, op, tensor):
            if op == self.expand_dims_op_1 and tensor != self.expand_dims_op_1.inputs[1]:
                return False
            elif op == self.expand_dims_op_2 and tensor != self.expand_dims_op_2.inputs[1]:
                return False
            return True

    class UnrolledTimeStepDescriptor(LayerDescriptor):
        def __init__(self, name, operations, cell_input_concat_op, gates_matmul_op, gates_biases_op, cell_output_op):
            super(LstmLayerResolver.UnrolledTimeStepDescriptor, self).__init__('LSTM', name, operations)
            self.cell_input_concat_op = cell_input_concat_op
            self.gates_matmul_op = gates_matmul_op
            self.gates_biases_op = gates_biases_op
            self.cell_output_op = cell_output_op
            self.cell_0 = self
            self.unrolled_cells = [self]
            self._is_stacked_cell = False

        def is_unrolled_cell_of(self, descriptor):
            return self.gates_matmul_op.inputs[1].op == descriptor.gates_matmul_op.inputs[1].op

        def is_cell_of_time_step_0(self):
            return self.cell_0 == self

        def time_steps(self):
            return len(self.unrolled_cells)

        def is_output_op(self, op):
            return op.outputs[0].name == self._output_tensor_name

        @property
        def output_names(self):
            if not self._is_stacked_cell:
                return [self._output_tensor_name]
            else:
                return [self.stacked_cell_output_name]

        def set_is_stacked_cell(self, is_stacked_cell):
            self._is_stacked_cell = is_stacked_cell

        @property
        def stacked_cell_output_name(self):
            return '{}_all_time_steps'.format(self._output_tensor_name)

        def get_output_names_for(self, input_tensors):
            """
            :type input_tensors: [tensorflow.Tensor]
            :rtype: [str]
            """
            if not self._is_stacked_cell:
                return super(LstmLayerResolver.UnrolledTimeStepDescriptor, self).get_output_names_for(input_tensors)
            else:
                return [self.stacked_cell_output_name]

        @property
        def _output_tensor_name(self):
            return str(self.unrolled_cells[-1].cell_output_op.outputs[0].name)

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(cell_sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            cell_input_concat_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat']
            gates_matmul_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul']
            gates_biases_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd']
            cell_output_op = match['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2']
            d = LstmLayerResolver.UnrolledTimeStepDescriptor(str(cell_output_op.name),
                                                             match.consumed_nodes,
                                                             cell_input_concat_op=cell_input_concat_op,
                                                             gates_matmul_op=gates_matmul_op,
                                                             gates_biases_op=gates_biases_op,
                                                             cell_output_op=cell_output_op)
            descriptors.append(d)

        matches = graph_matcher.match_sequence(state_sequence)
        for match in matches:
            state = match['rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros']
            expand_dims_op_1 = match['rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims']
            expand_dims_op_2 = match['rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims_2']
            d = LstmLayerResolver.StateDescriptor(str(state.name), match.consumed_nodes,
                                                  expand_dims_op_1, expand_dims_op_2)
            descriptors.append(d)

        if len(descriptors) == 0:
            return []

        return descriptors


class LstmLayerBuilder(LayerBuilder):
    _TENSORFLOW_INPUT_GATE_INDEX = 0
    _TENSORFLOW_FORGET_GATE_INDEX = 2
    _TENSORFLOW_OUTPUT_GATE_INDEX = 3
    _TENSORFLOW_STATE_GATE_INDEX = 1

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: LstmLayerResolver.UnrolledTimeStepDescriptor
        :rtype: int
        """
        if isinstance(descriptor, LstmLayerResolver.StateDescriptor):
            return

        input_descriptors = [d for d in input_descriptors if not isinstance(d, LstmLayerResolver.StateDescriptor)]
        if len(input_descriptors) not in [1, 3]:
            raise ConverterError('LSTM layer requires 1 or 3 inputs')

        input_shape = converter_context.graph_helper.get_op_output_shape(descriptor.cell_input_concat_op.inputs[0].op)
        state_shape = converter_context.graph_helper.get_op_output_shape(descriptor.cell_input_concat_op.inputs[1].op)

        gates_weights, input_weights = self._resolve_weights(descriptor, converter_context.graph_helper, state_shape)
        gates_biases = self._resolve_biases(descriptor, converter_context.graph_helper)

        def is_cell_input_descriptor(d):
            output_shape = []
            output_ops = [op for op in d.child_ops if d.is_output_op(op)]
            if len(output_ops) > 0:
                output_shape = converter_context.graph_helper.get_op_output_shape(output_ops[0])
            return len(output_shape) == 2 and output_shape[1] == descriptor.time_steps()

        cell_input_descriptors = list(filter(is_cell_input_descriptor, input_descriptors))
        cell_state_descriptors = [d for d in input_descriptors if d not in cell_input_descriptors]

        user_initial_state = len(list(cell_state_descriptors)) == 2

        is_stacked_above_cell = self.is_stacked_cell(input_descriptors)
        if not is_stacked_above_cell:
            if len(list(cell_input_descriptors)) != 1:
                raise ConverterError('Unable to resolve LSTM input layer name.')

            cell_input_name = cell_input_descriptors[0].output_names[0]
            input_layer_name = self._add_reshape_to_restore_time_dimension(
                converter_context, descriptor, cell_input_name, input_shape)
        else:
            input_layer_name = input_descriptors[0].output_names[0]

        is_stacked_below_cell = self.is_stacked_cell(output_descriptors)
        descriptor.set_is_stacked_cell(is_stacked_below_cell)

        output_names = [descriptor.stacked_cell_output_name]
        if user_initial_state or (not is_stacked_below_cell and len(output_descriptors) > 0):
            output_names.append('{}_state'.format(descriptor.output_names[0]))
            output_names.append(descriptor.output_names[0])

        h_0_input_name = cell_state_descriptors[0].output_names[0] if user_initial_state else ''
        c_0_input_name = cell_state_descriptors[1].output_names[0] if user_initial_state else ''
        return converter_context.model.add_lstm_layer(name=descriptor.output_names[0],
                                                      w_xc=input_weights,
                                                      b_c=gates_biases,
                                                      w_hc=gates_weights,
                                                      w_xc_static=None,
                                                      backward=False,
                                                      reset_state_at_time_step_0=not user_initial_state,
                                                      input_name=input_layer_name,
                                                      sequence_continuation_input_name='',
                                                      x_static_input_name='',
                                                      c_0_input_name=c_0_input_name,
                                                      h_0_input_name=h_0_input_name,
                                                      output_names=output_names
                                                      )

    @classmethod
    def is_stacked_cell(cls, descriptors):
        return len(descriptors) == 1 and isinstance(descriptors[0], LstmLayerResolver.UnrolledTimeStepDescriptor)

    @classmethod
    def _add_reshape_to_restore_time_dimension(cls, converter_context, descriptor, input_name, input_shape):
        reshape_layer_name = '{}_reshape'.format(descriptor.layer_name)
        reshape_output = [input_shape[0], descriptor.time_steps(), input_shape[1]]
        converter_context.model.add_reshape_layer(reshape_layer_name,
                                                  reshape_output,
                                                  input_name,
                                                  reshape_layer_name)
        return reshape_layer_name

    def _resolve_weights(self, descriptor, graph_helper, state_shape):
        merged_weights = graph_helper.evaluate_tensor_output(descriptor.gates_matmul_op.inputs[1])
        input_weights_slice_index = np.shape(merged_weights)[0] - state_shape[-1]
        weights_list = np.split(merged_weights,
                                indices_or_sections=[input_weights_slice_index],
                                axis=0)
        input_weights = weights_list[0]
        input_weights = self._reorder_tensorflow_gates_weights(input_weights)
        gates_weights = weights_list[1]
        gates_weights = self._reorder_tensorflow_gates_weights(gates_weights)
        return gates_weights, input_weights

    def _resolve_biases(self, descriptor, graph_helper):
        gates_biases = graph_helper.evaluate_tensor_output(descriptor.gates_biases_op.inputs[1])
        gates_biases = np.split(gates_biases, indices_or_sections=4, axis=0)
        self._add_scalar_to_gate_bias(self._TENSORFLOW_FORGET_GATE_INDEX, 1, gates_biases)
        gates_biases = self._reorder_tensorflow_gates_biases(gates_biases)
        return gates_biases

    @classmethod
    def _add_scalar_to_gate_bias(cls, gate_index, bias_value, gates_biases):
        gates_biases[gate_index] += bias_value

    @classmethod
    def _reorder_tensorflow_gates_weights(cls, weights):
        weights = np.split(weights, indices_or_sections=4, axis=1)
        reordered = [
            weights[cls._TENSORFLOW_INPUT_GATE_INDEX],
            weights[cls._TENSORFLOW_FORGET_GATE_INDEX],
            weights[cls._TENSORFLOW_OUTPUT_GATE_INDEX],
            weights[cls._TENSORFLOW_STATE_GATE_INDEX],
        ]
        return np.concatenate(reordered, axis=1)

    @classmethod
    def _reorder_tensorflow_gates_biases(cls, biases):
        reordered = [
            biases[cls._TENSORFLOW_INPUT_GATE_INDEX],
            biases[cls._TENSORFLOW_FORGET_GATE_INDEX],
            biases[cls._TENSORFLOW_OUTPUT_GATE_INDEX],
            biases[cls._TENSORFLOW_STATE_GATE_INDEX],
        ]
        return np.concatenate(reordered, axis=0)

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        if isinstance(descriptor, LstmLayerResolver.StateDescriptor):
            return

        self._merge_unrolled_input_cells(converter_context, input_descriptors, descriptor)
        if not descriptor.is_cell_of_time_step_0():
            descriptor.set_ignored(True)

    @classmethod
    def _merge_state_descriptor(cls, converter_context, descriptor, input_descriptors):
        lstm_state_inputs = [d for d in input_descriptors if isinstance(d, LstmLayerResolver.StateDescriptor)]
        for state in lstm_state_inputs:
            converter_context.merge_descriptors(state, descriptor.cell_0)
            state.set_ignored(False)

    @classmethod
    def _merge_unrolled_input_cells(cls, converter_context, input_descriptors, descriptor):
        lstm_inputs = [i for i in input_descriptors if isinstance(i, LstmLayerResolver.UnrolledTimeStepDescriptor)]
        unrolled_inputs = [d for d in lstm_inputs if descriptor.is_unrolled_cell_of(d.cell_0)]
        for input_descriptor in unrolled_inputs:
            converter_context.merge_descriptors(descriptor, input_descriptor.cell_0)
            input_descriptor.cell_0.unrolled_cells.append(descriptor)
            descriptor.cell_0 = input_descriptor.cell_0
