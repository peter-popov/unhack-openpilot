# =============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)

cell_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read', ['Identity']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat',
                          ['ConcatV2']),
    NonConsumableConverterSequenceNode('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read', ['Identity']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul',
                          ['MatMul']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd',
                          ['BiasAdd']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split', ['Split']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add', ['Add']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh', ['Tanh']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1', ['Sigmoid']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid', ['Sigmoid']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1', ['Mul']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul', ['Mul']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1', ['Add']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2', ['Sigmoid']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1', ['Tanh']),
    ConverterSequenceNode('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2', ['Mul']),
    NonConsumableConverterSequenceNode('stub_16', ['?']),
    NonConsumableConverterSequenceNode('stub_17', ['?']),
    NonConsumableConverterSequenceNode('stub_18', ['?']),
    NonConsumableConverterSequenceNode('stub_19', ['?']),
    NonConsumableConverterSequenceNode('stub_20', ['?']),
    NonConsumableConverterSequenceNode('stub_21', ['?']),
    NonConsumableConverterSequenceNode('stub_22', ['?']),
    NonConsumableConverterSequenceNode('stub_23', ['?']),
])
cell_sequence.set_inputs('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read', ['stub_16'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat',
                              ['stub_17', 'stub_18', 'stub_19'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split', 'stub_22'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/concat',
                               'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul',
                              ['stub_23', 'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/MatMul',
                               'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1',
                               'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul',
                               'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split',
                              ['stub_21',
                               'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/basic_lstm_cell/BiasAdd'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1',
                               'rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split'])
cell_sequence.set_inputs('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read', ['stub_20'])
cell_sequence.set_inputs('rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1',
                              ['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split'])
cell_sequence.set_outputs(['rnn/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2'])

state_sequence = GraphSequence([
    ConverterSequenceNode('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims', ['ExpandDims']),
    ConverterSequenceNode('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims_2', ['ExpandDims']),
    ConverterSequenceNode('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat', ['ConcatV2']),
    ConverterSequenceNode('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1', ['ConcatV2']),
    ConverterSequenceNode('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros', ['Fill']),
    ConverterSequenceNode('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1', ['Fill']),
    NonConsumableConverterSequenceNode('stub_6', ['?']),
    NonConsumableConverterSequenceNode('stub_7', ['?']),
    NonConsumableConverterSequenceNode('stub_8', ['?']),
    NonConsumableConverterSequenceNode('stub_9', ['?']),
    NonConsumableConverterSequenceNode('stub_10', ['?']),
    NonConsumableConverterSequenceNode('stub_11', ['?']),
    NonConsumableConverterSequenceNode('stub_12', ['?']),
    NonConsumableConverterSequenceNode('stub_13', ['?']),
    NonConsumableConverterSequenceNode('stub_14', ['?']),
    NonConsumableConverterSequenceNode('stub_15', ['?']),
])
state_sequence.set_inputs('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat', ['rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims','stub_10','stub_11'])
state_sequence.set_inputs('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims', ['stub_6','stub_7'])
state_sequence.set_inputs('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1', ['rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims_2','stub_12','stub_13'])
state_sequence.set_inputs('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims_2', ['stub_8','stub_9'])
state_sequence.set_inputs('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros', ['rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat','stub_14'])
state_sequence.set_inputs('rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1', ['rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1','stub_15'])
state_sequence.set_outputs(['rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1','rnn/MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros'])
