# =============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from snpe.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


real_div_sequence = GraphSequence([
    ConverterSequenceNode('root', ['RealDiv']),
    NonConsumableConverterSequenceNode('a', ['?']),
    NonConsumableConverterSequenceNode('b', ['?'])
])
real_div_sequence.set_inputs('root', ['a', 'b'])
real_div_sequence.set_outputs(['root'])

identity_sequence = GraphSequence([
    ConverterSequenceNode('root', ['Identity']),
    NonConsumableConverterSequenceNode('any', ['?']),
])
identity_sequence.set_inputs('root', ['any'])
identity_sequence.set_outputs(['root'])

placeholder_with_default_sequence = GraphSequence([
    ConverterSequenceNode('root', ['PlaceholderWithDefault']),
    NonConsumableConverterSequenceNode('any', ['?']),
])
placeholder_with_default_sequence.set_inputs('root', ['any'])
placeholder_with_default_sequence.set_outputs(['root'])

ignored_sequence_1 = GraphSequence([
    ConverterSequenceNode('root', ['Pack']),
    ConverterSequenceNode('a', ['Add']),
    ConverterSequenceNode('b', ['Add']),
    ConverterSequenceNode('c', ['Mul']),
    ConverterSequenceNode('d', ['Mul']),
    ConverterSequenceNode('e', ['?']),
    ConverterSequenceNode('f', ['?']),
    ConverterSequenceNode('g', ['?']),
    ConverterSequenceNode('h', ['?']),
    ConverterSequenceNode('i', ['?']),
    ConverterSequenceNode('j', ['?']),
    ConverterSequenceNode('k', ['?']),
    ConverterSequenceNode('l', ['?'])
])
ignored_sequence_1.set_inputs('root', ['a', 'b', 'e', 'f'])
ignored_sequence_1.set_inputs('a', ['c', 'g'])
ignored_sequence_1.set_inputs('b', ['d', 'h'])
ignored_sequence_1.set_inputs('c', ['i', 'j'])
ignored_sequence_1.set_inputs('d', ['k', 'l'])
ignored_sequence_1.set_outputs(['root'])

ignored_sequence_2 = GraphSequence([
    ConverterSequenceNode('root', ['Pack']),
    ConverterSequenceNode('a', ['Mul']),
    ConverterSequenceNode('b', ['Mul']),
    ConverterSequenceNode('e', ['?']),
    ConverterSequenceNode('f', ['?'])
])
ignored_sequence_2.set_inputs('root', ['a', 'b', 'e', 'f'])
ignored_sequence_2.set_outputs(['root'])

dropout_cell_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('is_training/read', ['?']),
    ConverterSequenceNode('Dropout/cond/Switch', ['Switch']),
    NonConsumableConverterSequenceNode('Dropout/cond/switch_t', ['Identity']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/random_uniform/min', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/random_uniform/max', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/Shape', ['Const']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/sub', ['Sub']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/RandomUniform', ['RandomUniform']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/mul', ['Mul']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform', ['Add']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/keep_prob', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/pred_id', ['Identity']),
    ConverterSequenceNode('Dropout/cond/dropout/add', ['Add']),
    ConverterSequenceNode('Dropout/cond/dropout/div/Switch', ['Switch']),
    ConverterSequenceNode('Dropout/cond/dropout/Floor', ['Floor']),
    ConverterSequenceNode('Dropout/cond/dropout/div', ['RealDiv']),
    ConverterSequenceNode('Dropout/cond/dropout/mul', ['Mul']),
    ConverterSequenceNode('Dropout/cond/Switch_1', ['Switch']),
    ConverterSequenceNode('Dropout/cond/Merge', ['Merge']),
    NonConsumableConverterSequenceNode('stub_20', ['?']),
    NonConsumableConverterSequenceNode('stub_25', ['?']),
])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/add',
                              ['Dropout/cond/dropout/keep_prob', 'Dropout/cond/dropout/random_uniform'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/Floor', ['Dropout/cond/dropout/add'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/mul',
                              ['Dropout/cond/dropout/random_uniform/RandomUniform',
                               'Dropout/cond/dropout/random_uniform/sub'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/div',
                              ['Dropout/cond/dropout/div/Switch', 'Dropout/cond/dropout/keep_prob'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform', ['Dropout/cond/dropout/random_uniform/mul',
                                                                      'Dropout/cond/dropout/random_uniform/min'])
dropout_cell_sequence.set_inputs('Dropout/cond/Switch', ['stub_20', 'is_training/read'])
dropout_cell_sequence.set_inputs('Dropout/cond/pred_id', ['is_training/read'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/RandomUniform',
                              ['Dropout/cond/dropout/Shape'])
dropout_cell_sequence.set_inputs('Dropout/cond/Merge', ['Dropout/cond/Switch_1', 'Dropout/cond/dropout/mul'])
dropout_cell_sequence.set_inputs('Dropout/cond/switch_t', ['Dropout/cond/Switch'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/mul',
                              ['Dropout/cond/dropout/div', 'Dropout/cond/dropout/Floor'])
dropout_cell_sequence.set_inputs('Dropout/cond/Switch_1', ['stub_25', 'Dropout/cond/pred_id'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/sub',
                              ['Dropout/cond/dropout/random_uniform/max',
                               'Dropout/cond/dropout/random_uniform/min'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/div/Switch', ['stub_25', 'Dropout/cond/pred_id'])
dropout_cell_sequence.set_outputs(['Dropout/cond/Merge'])

batchnorm_fold_sequence = GraphSequence([
    ConverterSequenceNode('add', ['Add']),
    ConverterSequenceNode('rsqrt', ['Rsqrt']),
    ConverterSequenceNode('mul', ['Mul']),
    ConverterSequenceNode('mul_1', ['Mul']),
    ConverterSequenceNode('sub', ['Sub']),
    ConverterSequenceNode('mul_fold', ['Mul']),
    NonConsumableConverterSequenceNode('fake_quant', ['FakeQuantWithMinMaxVars']),  # Output
    ConverterSequenceNode('mean', ['?']),
    ConverterSequenceNode('beta', ['?']),
    ConverterSequenceNode('variance', ['?']),
    ConverterSequenceNode('epsilon', ['?']),
    ConverterSequenceNode('gamma', ['?']),
    ConverterSequenceNode('weights', ['?']),
    NonConsumableConverterSequenceNode('min', ['?']),
    NonConsumableConverterSequenceNode('max', ['?'])
])
batchnorm_fold_sequence.set_inputs('add', ['variance', 'epsilon'])
batchnorm_fold_sequence.set_inputs('rsqrt', ['add'])
batchnorm_fold_sequence.set_inputs('mul', ['rsqrt', 'gamma'])
batchnorm_fold_sequence.set_inputs('mul_fold', ['weights', 'mul'])
batchnorm_fold_sequence.set_inputs('mul_1', ['mul', 'mean'])
batchnorm_fold_sequence.set_inputs('sub', ['mul_1', 'beta'])
batchnorm_fold_sequence.set_inputs('fake_quant', ['min', 'max', 'mul_fold'])
batchnorm_fold_sequence.set_outputs(['sub', 'fake_quant'])

batchnorm_fold_sequence_reshape = GraphSequence([
    ConverterSequenceNode('add', ['Add']),
    ConverterSequenceNode('rsqrt', ['Rsqrt']),
    ConverterSequenceNode('mul', ['Mul']),
    ConverterSequenceNode('mul_1', ['Mul']),
    ConverterSequenceNode('sub', ['Sub']),
    ConverterSequenceNode('reshape', ['Reshape']),
    ConverterSequenceNode('mul_fold', ['Mul']),
    NonConsumableConverterSequenceNode('fake_quant', ['FakeQuantWithMinMaxVars']),  # Output
    ConverterSequenceNode('mean', ['?']),
    ConverterSequenceNode('beta', ['?']),
    ConverterSequenceNode('variance', ['?']),
    ConverterSequenceNode('epsilon', ['?']),
    ConverterSequenceNode('gamma', ['?']),
    ConverterSequenceNode('weights', ['?']),
    ConverterSequenceNode('shape', ['?']),
    NonConsumableConverterSequenceNode('min', ['?']),
    NonConsumableConverterSequenceNode('max', ['?'])
])
batchnorm_fold_sequence_reshape.set_inputs('add', ['variance', 'epsilon'])
batchnorm_fold_sequence_reshape.set_inputs('rsqrt', ['add'])
batchnorm_fold_sequence_reshape.set_inputs('mul', ['rsqrt', 'gamma'])
batchnorm_fold_sequence_reshape.set_inputs('reshape', ['mul', 'shape'])
batchnorm_fold_sequence_reshape.set_inputs('mul_fold', ['weights', 'reshape'])
batchnorm_fold_sequence_reshape.set_inputs('mul_1', ['mul', 'mean'])
batchnorm_fold_sequence_reshape.set_inputs('sub', ['mul_1', 'beta'])
batchnorm_fold_sequence_reshape.set_inputs('fake_quant', ['min', 'max', 'mul_fold'])
batchnorm_fold_sequence_reshape.set_outputs(['sub', 'fake_quant'])
