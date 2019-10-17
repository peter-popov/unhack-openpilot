#!/usr/bin/env python
# =============================================================================
#
#  Copyright (c) 2015-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from collections import OrderedDict
import sys
import tensorflow as tf
from tensorflow.python.framework.errors import InvalidArgumentError

try:
    from snpe.dlc_utils import modeltools
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)
from snpe.converters.common.utils import code_to_message


class GraphVisitor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_operation(self, node_def):
        """
        :type node_def: tensorflow.NodeDef
        :rtype: None
        """
        pass

    @abstractmethod
    def visit_scope(self, scope, nodes_defs):
        """
        :type scope: str
        :type nodes_defs: list[tensorflow.NodeDef]
        :return: None
        """
        pass


class VisitableGraph(object):
    def __init__(self, ops):
        """
        :type ops: list[tensorflow.Operation]
        """
        self._operations = ops

    def accept(self, visitor):
        """
        Walks the graph and calls the specified visitor for each node and scope in the graph.
        :type visitor GraphVisitor
        :rtype: None
        """
        for op in self._operations:
            visitor.visit_operation(op)

        scopes = self._scopes_for_nodes(self._operations)
        for scope, ops in list(scopes.items()):
            visitor.visit_scope(scope, ops)

    @classmethod
    def _scopes_for_nodes(cls, ops):
        """
        :type ops: list[tensorflow.Operation]
        :rtype: dict[str,tensorflow.Operation]
        """
        scope_nodes_map = OrderedDict()
        for op in ops:
            splits = op.name.split('/')
            scope = '/'.join(splits[:-1]) if len(splits) > 1 else str(op.name)
            nodes = scope_nodes_map.get(op.name, [])
            if len(nodes) == 0 or len([node for node in nodes if node.type != 'Const']) == 0:
                nodes = scope_nodes_map.get(scope, [])
            else:
                scope = str(op.name)
            nodes.append(op)
            scope_nodes_map[scope] = nodes
        return scope_nodes_map


class GraphPrinter(GraphVisitor, object):
    def visit_operation(self, op):
        pass

    def visit_scope(self, scope, ops):
        logging.debug(code_to_message.get_debugging_message('DEBUG_TF_SCOPE_PRINT')(scope))
        for op in ops:
            logging.debug(code_to_message.get_debugging_message('DEBUG_TF_OP_NAME_TYPE_PRINT')(op.name, op.type))


class GraphHelper(object):
    def __init__(self, session, model, ops):
        """
        Provides several helper methods to navigate the Tensorflow Graph.
        :type session: tensorflow.Session
        :type model: converters.tensorflow.loader.Model
        :type: ops: list[tensorflow.Operation]
        """
        self._session = session
        self._model = model
        self._graph = session.graph
        self._op_output_map = dict()
        self._tensor_shape_cache = dict()  # type: dict(str, list[int])
        self._tensor_value_cache = dict()  # type: dict(str, np.ndarray)
        self._placeholders_stubs_map = dict()
        if self._model is not None:
            input_names = [graph_input.name for graph_input in self._model.inputs]
            self._placeholders_stubs_map = self._create_placeholders_tensors(session, input_names)
        self._op_output_map = self._map_operations_outputs(ops)
        self._evaluate_tensor_shapes(ops)

    @classmethod
    def _create_placeholders_tensors(cls, session, inputs):
        placeholders_stubs_map = dict()
        # run in isolated session so that the memory gets cleared out after retrieving the output
        with tf.Session(graph=session.graph) as sess:
            for op in sess.graph.get_operations():
                if op.type == 'Placeholder' and op.name not in inputs:
                    dtype = np.float32
                    if op.get_attr('dtype') == tf.uint8:
                        dtype = np.uint8

                    tensor = sess.graph.get_tensor_by_name(GraphHelper.indexed_tensor_name(op.name))
                    shape = tensor.get_shape().as_list()
                    shape = [d if d is not None else 1 for d in shape]
                    tensor = np.zeros(shape, dtype=dtype)
                    placeholders_stubs_map[str(op.name)] = tensor
            return placeholders_stubs_map

    def dump(self):
        for n, s in list(self._tensor_shape_cache.items()):
            print(n, s)

    @classmethod
    def _map_operations_outputs(cls, operations):
        """
        :type operations: list[tensorflow.Operation]
        :rtype: dict[tensorflow.Operation, list[tensorflow.Operation]]
        """
        visitable_graph = VisitableGraph(operations)
        mapper = OutputOperationsMapper()
        visitable_graph.accept(mapper)
        return mapper.output_ops_map

    def get_tensor_by_name(self, tensor_name):
        """
        :type tensor_name: str
        :rtype: tensorflow.Tensor
        """
        return self._graph.get_tensor_by_name(self.indexed_tensor_name(tensor_name))

    @classmethod
    def indexed_tensor_name(cls, name, tensor_index=0):
        """
        :type name: str
        :type tensor_index: int
        :rtype: str
        """
        return '{}:{}'.format(name, tensor_index) if ':' not in name else name

    def get_op_output_shape(self, operation, tensor_index=0):
        """
        :type operation: tensorflow.Operation
        :type tensor_index: int
        :rtype: list[int]
        """
        tensor = self._graph.get_tensor_by_name(
            GraphHelper.indexed_tensor_name(operation.name, tensor_index))
        if tensor.name not in self._tensor_shape_cache:
            shape = self._get_tensor_output_shape(tensor)
            if len(shape) == 0:
                shapes = self._evaluate_tensors_output_shape([tensor])
                shape = shapes[tensor]
        else:
            shape = self._tensor_shape_cache[tensor.name]
        return shape

    def _evaluate_tensors_output_shape(self, tensors):
        """
        :type tensor: list(tensorflow.Tensor)
        :return: dict[tensorflow.Tensor, list[int]]
        """
        shapes_map = dict()
        outputs_map = self.evaluate_tensors_output(tensors)
        for tensor, output in list(outputs_map.items()):
            shape = list(np.shape(output))
            self._tensor_shape_cache[tensor.name] = shape
            shapes_map[tensor] = shape

        return shapes_map

    def evaluate_tensor_output(self, tensor):
        """
        :type tensor: tensorflow.Tensor
        :return: np.ndarray
        """
        outputs_map = self.evaluate_tensors_output([tensor])
        return outputs_map[tensor]

    def evaluate_tensors_output(self, tensors):
        """
        :type tensors: list(tensorflow.Tensor)
        :return: dict(tensorflow.Tensor, np.ndarray)
        """
        ignore_batch = True
        for t in tensors:
            if t.op.type != 'Const':
                ignore_batch = False
                break

        input_tensors = dict()
        for i in self._model.inputs:
            indexed_tensor_name = GraphHelper.indexed_tensor_name(i.name)
            if ignore_batch and str(self.get_tensor_by_name(indexed_tensor_name).shape[0]) == '?':
                input_tensors[indexed_tensor_name] = np.zeros([1] + i.shape[1:], dtype=np.float)
            else:
                input_tensors[indexed_tensor_name] = np.zeros(i.shape, dtype=np.float)

        for name, tensor in list(self._placeholders_stubs_map.items()):
            indexed_tensor_name = GraphHelper.indexed_tensor_name(name)
            if ignore_batch:
                input_tensors[indexed_tensor_name] = [tensor[0]]
            else:
                input_tensors[indexed_tensor_name] = tensor
        outputs_map = dict()
        requiring_evaluation = []
        for t in tensors:
            if t.name in self._tensor_value_cache:
                outputs_map[t] = self._tensor_value_cache[t.name]
            else:
                requiring_evaluation.append(t)

        if len(requiring_evaluation) > 0:
            try:
                # run in isolated session so that the memory gets cleared out after retrieving the output
                with tf.Session(graph=self._graph) as sess:
                    outputs = sess.run(fetches=requiring_evaluation, feed_dict=input_tensors)
                outputs = dict(list(zip(requiring_evaluation, outputs)))
                for t, o in list(outputs.items()):
                    self._tensor_value_cache[t.name] = o
                outputs_map.update(outputs)
                requiring_evaluation = []
            except InvalidArgumentError:
                pass

        for t in requiring_evaluation:
            try:
                # run in isolated session so that the memory gets cleared out after retrieving the output
                with tf.Session(graph=self._graph) as sess:
                    outputs = sess.run(fetches=[t], feed_dict=input_tensors)
                # outputs = self._session.run(fetches=[t], feed_dict=input_tensors)
                self._tensor_value_cache[t.name] = outputs[0]
                outputs_map[t] = outputs[0]
            except InvalidArgumentError:
                shape = (1,)
                try:
                    tensor_shape = t.get_shape().as_list()
                    if tensor_shape and None not in tensor_shape:
                        shape = tensor_shape
                except Exception:
                    pass

                outputs_map[t] = np.zeros(shape, dtype=np.float32)
        return outputs_map

    @classmethod
    def _get_tensor_output_shape(cls, tensor):
        """
        :type tensor: tensorflow.Tensor
        :rtype: list[int]
        """
        shape = []
        if tensor.get_shape():
            tensor_shape = [dim if dim else -1 for dim in tensor.get_shape().as_list()]
            if len(tensor_shape) > 0 and not cls._has_unresolved_dimension(tensor_shape):
                shape = cls._with_single_batch_dimension(tensor_shape)

        return shape

    @classmethod
    def _with_single_batch_dimension(cls, shape):
        """
        :type shape: list[int]
        :rtype: list[int]
        """
        copy = list(shape)
        if copy[0] == -1:
            copy[0] = 1
        return copy

    @classmethod
    def _has_unresolved_dimension(cls, shape):
        """
        :type shape: list[int]
        :rtype: bool
        """
        return len(shape) > 0 and -1 in shape

    @classmethod
    def filter_ops_by_type(cls, operations, operation_type):
        """
        :type operations: list[tensorflow.Operation]
        :type operation_type: str
        :rtype: list[tensorflow.Operation]
        """
        return [operation for operation in operations if operation.type.upper() == operation_type.upper()]

    @classmethod
    def filter_op_by_type(cls, operations, operation_type):
        """
        :type operations: list[tensorflow.Operation]
        :type operation_type: str
        :rtype: tensorflow.Operation
        """
        ops = cls.filter_ops_by_type(operations, operation_type)
        if len(ops) == 0:
            raise OperationNotFoundError()
        return ops[0]

    @classmethod
    def filter_single_op_by_type(cls, operations, operation_type):
        ops = cls.filter_ops_by_type(operations, operation_type)
        if len(ops) == 0:
            operations_message = [(op.name, op.type) for op in operations]
            raise OperationNotFoundError(
                code_to_message.get_error_message('ERROR_TF_OPERATION_NOT_FOUND')(operation_type, operations_message))
        if len(ops) > 1:
            raise OperationNotFoundError(
                code_to_message.get_error_message('ERROR_TF_MULTIPLE_NODES_FOUND')(operation_type))
        return ops[0]

    def get_op_outputs(self, operation):
        return self._op_output_map.get(operation, [])

    @classmethod
    def get_op_input_tensors(cls, operations, input_types):
        """
        :type operations: tensorflow.Operation
        :type input_types:
        :return: tuple[tensorflow.Tensor]
        """
        tensors = [tensor for tensor in operations.inputs]
        types = [t.op.type for t in tensors]
        if len(types) != len(input_types):
            raise TensorNotFoundError(
                code_to_message.get_error_message('ERROR_TF_INPUT_DOES_NOT_MATCH_COUNT')(operations.name, types, input_types))

        input_tensors = []
        for i, t in enumerate(tensors):
            if types[i] == input_types[i] or input_types[i] == '?':
                input_tensors.append(t)
            else:
                raise TensorNotFoundError(
                    code_to_message.get_error_message('ERROR_TF_INPUT_DOES_NOT_MATCH_TYPES')(operations.name,
                                                                                             types,
                                                                                             input_types))

        if len(input_tensors) > 1:
            return tuple(input_tensors)
        else:
            return input_tensors[0]

    @classmethod
    def get_op_sequence(cls, operation, types):
        """
        :type operation: tensorflow.Operation
        :type types: list[str]
        :rtype: list[tensorflow.Operation]
        """
        result = []
        if len(types) == 0 or operation.type != types[0]:
            raise OperationNotFoundError()

        result.append(operation)

        if len(types[1:]) > 0:
            matches = [t.op for t in operation.inputs if t.op.type == types[1]]
            if len(matches) == 1:
                result += cls.get_op_sequence(matches[0], types[1:])
            else:
                raise OperationNotFoundError()
        return result

    def _evaluate_tensor_shapes(self, ops):
        """
        :type ops: list(tensorflow.Operation)
        :rtype: None
        """
        tensors = set()
        for t in [t for op in ops for t in op.outputs]:
            tensors.add(t)

        for t in [t for op in ops for t in op.inputs]:
            tensors.add(t)

        try:
            self._evaluate_tensors_output_shape(tensors)
        except Exception:
            # If we can't evaluate the graph ops in one pass
            # fallback to on-demand evaluation later
            logger = logging.getLogger()
            logger.warning(code_to_message.get_error_message('ERROR_TF_FALLBACK_TO_ONDEMAND_EVALUATION'))

    @classmethod
    def check_tensor_const_origin(cls, tensor):
        queue = [tensor.op]
        visited = []

        while queue:
            head = queue.pop()

            if head in visited:
                continue

            for input_op in head.inputs:
                if input_op.op.type == 'Placeholder':
                    return False

                queue.append(input_op.op)

        return True


class ConverterError(Exception):
    """
    Defines a generic error for any converter errors.
    """
    pass


class OperationNotFoundError(LookupError):
    """
    Defines an error for when a required operation is not found by a method.
    """
    pass


class TensorNotFoundError(LookupError):
    """
    Defines an error for when a required operation is not found by a method.
    """
    pass


def scoped_op_name(scope_name, operation):
    """
    :type scope_name: str
    :type operation: tensorflow.Operation
    :rtype: str
    """
    op_name = str(operation.name)
    if scope_name == op_name:
        return "{}/{}".format(scope_name, op_name.split('/')[-1])
    else:
        return op_name


def uniques(values):
    """
    :type values: list
    :rtype: list
    """
    dictionary = OrderedDict()
    for v in values:
        if v not in dictionary:
            dictionary[v] = v
    return list(dictionary.keys())


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


class OutputOperationsMapper(GraphVisitor, object):
    def __init__(self):
        super(OutputOperationsMapper, self).__init__()
        self.output_ops_map = OrderedDict()  # type: dict[tf.Operation,list[tf.Operation]]

    def visit_scope(self, scope, ops):
        pass

    def visit_operation(self, op):
        """
        :type op: tensorflow.Operation
        :rtype: None
        """
        for t in op.inputs:
            if t.op not in self.output_ops_map:
                self.output_ops_map[t.op] = []
            self.output_ops_map[t.op].append(op)


class OperationExecutionSorter(object):
    def __init__(self, ops):
        self.input_ops = []
        self.output_ops = []
        self.ops_map = dict()
        for op in ops:
            op_wrapper = OperationExecutionSorter.OpWrapper(op)
            self.ops_map[op_wrapper.name] = op_wrapper
        self._connect_wrapped_ops()

    class OpWrapper:
        def __init__(self, tf_op):
            self.tf_op = tf_op
            self.name = str(tf_op.name)
            self.order = -1
            self.outputs = []

    def _connect_wrapped_ops(self):
        for op in list(self.ops_map.values()):
            for input_tensor in op.tf_op.inputs:
                if input_tensor.op.name not in self.ops_map:
                    continue
                input_op = self.ops_map[input_tensor.op.name]
                input_op.outputs.append(op)

    def sort(self, input_ops_names, output_ops_names):
        self._prepare_inputs_and_outputs(input_ops_names, output_ops_names)
        for input_op in self.input_ops:
            self._resolve_ops_in_execution_order(input_op, self.output_ops)

        self._flag_unvisited_nodes()

        sorted_in_execution_order = sorted(list(self.ops_map.values()), cmp=lambda a, b: a.order - b.order)
        return [op.tf_op for op in sorted_in_execution_order]

    def _prepare_inputs_and_outputs(self, input_ops_names, output_ops_names):
        self.input_ops = []
        self.output_ops = []
        for op_wrapper in list(self.ops_map.values()):
            if op_wrapper.name in input_ops_names:
                op_wrapper.order = 0
                self.input_ops.append(op_wrapper)
            elif op_wrapper.name in output_ops_names:
                self.output_ops.append(op_wrapper)

    @classmethod
    def _resolve_ops_in_execution_order(cls, input_op, output_ops):
        queue = [input_op]
        while len(queue) > 0:
            current_op = queue.pop(0)
            if current_op in output_ops:
                continue

            for output_op in reversed(current_op.outputs):
                output_order = max(output_op.order, current_op.order + 1)
                if output_order > output_op.order:
                    output_op.order = output_order
                    queue.insert(0, output_op)

    def _flag_unvisited_nodes(self):
        for op in list(self.ops_map.values()):
            if op.order == -1:
                op.order = sys.maxsize
