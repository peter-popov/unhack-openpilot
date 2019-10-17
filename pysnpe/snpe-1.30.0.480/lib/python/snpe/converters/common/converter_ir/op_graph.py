# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import inspect

from snpe.converters.common.converter_ir import op_adapter
from snpe.converters.common.converter_ir.axis_tracker import AxisTracker
from snpe.converters.common.utils.snpe_converter_utils import *
from snpe.converters.common.utils.code_to_message import *


class OpNode(object):
    def __init__(self, op, input_names, output_names):
        self.op = op
        self.input_names = input_names
        self.output_names = output_names


class Buffer(object):

    def __init__(self, name, shape, producer):
        self.name = name
        self.producer = producer
        self.consumers = set()
        self.shape = shape
        self.axis_format = AxisTracker.AxisFormat.NOT_YET_DEFINED

    def rank(self):
        return len(self.shape)

    def get_buf_dims(self):
        return self.shape

    def get_axis_order(self):
        """Translate AxisFormat enum to modeltools axis order list"""
        if self.axis_format == 'NSC':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        if self.axis_format == 'NCS':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        elif self.axis_format == 'FEATURE':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'BTF':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.TIME,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'NONTRIVIAL':
            return [AxisTracker.AxisAnnotations.NONTRIVIAL]
        else:
            raise ValueError("Encountered unexpected axis format for get_axis_order: %s" % self.axis_format)


class BufferCriteria(object):
    """
    Class(enum) to use for setting buffer criteria on inputs/outputs for validating matched node sequences
    """
    # to be used for individual buffers
    ALL = "ALL"  # all the buffer(s) must be this same expected op_type
    ANY = "ANY"  # There can be one or more of this op_type as buffer(s)
    NONE = "NONE"  # None of the buffer(s) should be of this type

    # to be used for set of buffers
    MATCH_NUM_BUFS = "MATCH_NUM_BUFS"  # the expected number of buffers must be same length as matched buffers
    FLEXIBLE_NUM_BUFS = "FLEXIBLE_NUM_BUFS"  # the expected number of buffers doesnt need to be equal to matched buffers


class InputType(object):
    """
    Contains supported input types. This will be used by DSP to determine quantization
    """
    IMAGE = "image"  # input is float between 0-255 and the input's mean is 0.0f and the input's max is 255.0f
    DEFAULT = "default"  # pass the input as floats to the dsp directly and the DSP will quantize it
    OPAQUE = "opaque"  # assumes input is float because the consumer layer(i.e next layer) requires it as float,
    # therefore it won't be quantized by DSP

    @classmethod
    def get_supported_types(cls):
        return [cls.IMAGE, cls.DEFAULT, cls.OPAQUE]

    @classmethod
    def is_valid_type(cls, input_type):
        return input_type in cls.get_supported_types()


class InputEncodings(object):
    """
    Contains supported input encodings
    """
    BGR = "bgr"
    RGB = "rgb"
    RGBA = "rgba"
    ARGB32 = "argb32"
    NV21 = "nv21"
    TIME_SERIES = "time_series"
    OTHER = "other"

    @classmethod
    def get_supported_encodings(cls):
        return [cls.BGR, cls.RGB, cls.RGBA, cls.ARGB32, cls.NV21, cls.TIME_SERIES, cls.OTHER]

    @classmethod
    def is_valid_encoding(cls, input_encoding):
        return input_encoding in cls.get_supported_encodings()


class QuantParams(object):
    """
    Contains supported quantization params
    """
    BN_PARAMS = "bn_params"
    OUTPUT_ENCODINGS = "output_encodings"
    PARAM_ENCODINGS = "param_encodings"

    @classmethod
    def get_supported_quant_params(cls):
        return [cls.BN_PARAMS, cls.OUTPUT_ENCODINGS, cls.PARAM_ENCODINGS]

    @classmethod
    def is_valid_quant_param(cls, input_encoding):
        return input_encoding in cls.get_supported_quant_params()


class IROpGraph(object):
    def __init__(self, naming_policy, shape_inference_policy, input_types, input_encodings):
        self.naming_policy = naming_policy
        self.shape_inference_policy = shape_inference_policy
        self.inputs_type_dict = self._create_input_types_dict(input_types)
        self.inputs_encoding_dict = self._create_input_encodings_dict(input_encodings)
        self.nodes_by_name = {}
        self.nodes_in_order = []
        self.buffers = {}
        self.quantization_params = {}

    def __iter__(self):
        return iter(self.nodes_in_order)

    @staticmethod
    def _create_input_types_dict(input_types):
        log_assert(all(InputType.is_valid_type(type_) for _, type_ in input_types),
                   get_error_message("ERROR_UNSUPPORTED_INPUT_TYPE")(InputType.get_supported_types()))
        return {input_name: input_type for input_name, input_type in input_types}

    @staticmethod
    def _create_input_encodings_dict(input_encodings):
        log_assert(all(InputEncodings.is_valid_encoding(encoding) for _, encoding in input_encodings),
                   get_error_message("ERROR_UNSUPPORTED_INPUT_ENCODING")(InputEncodings.get_supported_encodings()))
        return {input_name: input_encoding for input_name, input_encoding in input_encodings}

    def get_input_type(self, input_name):
        # use input_type: default as the default for all inputs
        return self.inputs_type_dict.get(input_name, InputType.DEFAULT)

    def get_input_encoding(self, input_name):
        # use input_encoding: bgr as the default for all inputs
        return self.inputs_encoding_dict.get(input_name, InputEncodings.BGR)

    def add_quantization_params(self, op_name, **kwargs):
        log_assert(all(QuantParams.is_valid_quant_param(param) for param, _ in kwargs.iteritems()),
                   get_error_message("ERROR_UNSUPPORTED_QUANT_PARAM")(QuantParams.get_supported_quant_params()))

        self.quantization_params.update({op_name: {
            QuantParams.BN_PARAMS: kwargs.get(QuantParams.BN_PARAMS, {}),
            QuantParams.OUTPUT_ENCODINGS: kwargs.get(QuantParams.OUTPUT_ENCODINGS, {}),
            QuantParams.PARAM_ENCODINGS: kwargs.get(QuantParams.PARAM_ENCODINGS, {})
        }})

    def __insert_node(self, node, output_shapes, idx=-1):
        """Insert a node into the graph's internal data structures.

        node: Node to be inserted
        output_shapes: shapes of the node's output buffers, which must be created.
        idx: index in nodes_in_order at which to insert. By default, appends to
             the list."""
        for name, shape in zip(node.output_names, output_shapes):
            self.buffers[name] = Buffer(name, shape, node)

        for name in node.input_names:
            self.buffers[name].consumers.add(node)

        self.nodes_by_name[node.op.name] = node
        if idx == -1:
            self.nodes_in_order.append(node)
        else:
            self.nodes_in_order.insert(idx, node)

    def add(self, op, input_names, output_names):
        """
        Adds op to graph by creating a node and corresponding buffer, as well as update
        input and output buffers for node.
        :param op: an operation from op_adapter class
        :param input_names: inputs to node. (This will be the buffer input names)
        :param output_names: output buffer names of node
        :return: The created node for op.
        """
        op.name = self.naming_policy.get_op_name(op)

        if not isinstance(input_names, list):
            input_names = [input_names]
        input_names = self.naming_policy.get_input_names(op, input_names)

        input_shapes = []
        for name in input_names:
            if name not in self.buffers:
                raise KeyError("Graph has no buffer %s, referred to as input for %s" % (name, op.name))
            input_shapes.append(self.buffers[name].shape)

        if not isinstance(output_names, list):
            output_names = [output_names]
        output_names = self.naming_policy.get_output_names(op, output_names)

        node = OpNode(op, input_names, output_names)

        output_shapes = self.shape_inference_policy.infer_shape(op, input_shapes)
        if len(output_shapes) != len(output_names):
            raise ValueError("Op %s: produced %d output shapes, but have %d outputs" % (op.name, len(output_shapes),
                                                                                        len(output_names)))

        # at this point everything should be error free, so it's fine to actually
        # touch the data structures
        self.__insert_node(node, output_shapes)

        # return the added node
        return node

    def replace(self, old_op, new_op):
        old_node = self.nodes_by_name[old_op.name]
        input_buffers = self.get_input_buffers(old_node)
        output_buffers = self.get_output_buffers(old_node)
        input_names = [buf.name for buf in input_buffers]
        output_names = [buf.name for buf in output_buffers]

        # Create OpNode for the new op
        new_op.name = self.naming_policy.get_op_name(new_op)
        new_node = OpNode(new_op, input_names, output_names)

        # Replace the op in buffers
        input_shapes = []
        for buf in input_buffers:
            buf.consumers.remove(old_node)
            buf.consumers.add(new_node)
            input_shapes.append(buf.shape)

        output_shapes = self.shape_inference_policy.infer_shape(new_op, input_shapes)
        for i, buf in enumerate(output_buffers):
            buf.producer = new_node
            buf.shape = output_shapes[i]

        # Replace the op in op-lists
        idx = self.nodes_in_order.index(old_node)
        self.nodes_by_name[new_op.name] = new_node
        if idx == -1:
            self.nodes_in_order.append(new_node)
        else:
            self.nodes_in_order.insert(idx, new_node)

        del self.nodes_by_name[old_node.op.name]
        self.nodes_in_order.remove(old_node)

    def add_input(self, name, shape):
        input_type = self.get_input_type(name)
        input_encoding = self.get_input_encoding(name)
        op = op_adapter.InputOp(name, shape,
                                input_encoding_in=input_encoding,
                                input_encoding_out=InputEncodings.BGR,  # always default to BGR
                                input_type=input_type)
        output_names = self.naming_policy.get_output_names(op, [name])

        node = OpNode(op, [], output_names)
        self.__insert_node(node, [shape])

        # return the added input node
        return node

    def inject(self, op, input_name, output_name, consumer_names=None):
        op.name = self.naming_policy.get_op_name(op)
        if input_name not in self.buffers:
            raise KeyError("Cannot inject op %s onto nonexistent buffer %s" % (op.name, input_name))

        input_buffer = self.buffers[input_name]
        if consumer_names is None:
            old_consumers = list(input_buffer.consumers)
            input_buffer.consumers.clear()
        else:
            old_consumers = []
            for name in consumer_names:
                if name not in self.nodes_by_name:
                    raise KeyError("Cannot inject op %s with nonexistent consumer %s" % (op.name, name))
                consumer = self.nodes_by_name[name]
                if consumer not in input_buffer.consumers:
                    raise KeyError("Cannot inject op %s, specified consumer %s does not actually consume input"
                                   " buffer %s" % (op.name, name, input_name))

                old_consumers.append(consumer)
                input_buffer.consumers.remove(consumer)

        output_name = self.naming_policy.get_output_names(op, [output_name])[0]
        producer_idx = self.nodes_in_order.index(input_buffer.producer)
        output_shapes = self.shape_inference_policy.infer_shape(op, [input_buffer.shape])
        node = OpNode(op, [input_name], [output_name])
        self.__insert_node(node, output_shapes, producer_idx+1)

        output_buffer = self.buffers[output_name]
        for consumer in old_consumers:
            output_buffer.consumers.add(consumer)
            for i, name in enumerate(consumer.input_names):
                if name == input_name:
                    consumer.input_names[i] = output_name

    def prune(self, node):
        """Remove a node and its output buffers from the graph completely.
        Will raise an exception if the node has any successors."""

        output_buffers = self.get_output_buffers(node)
        consumers = []
        for buf in output_buffers:
            consumers.extend(buf.consumers)
        consumers = [c.op.name for c in consumers]
        if len(consumers) > 0:
            raise RuntimeError("Cannot prune node %s, which has the following successors: %s"
                               % (node.op.name, consumers))

        for buf in output_buffers:
            del self.buffers[buf.name]
        # loop through as set to support scenarios where a node is listed as input more than once
        for buf in set(self.get_input_buffers(node)):
            buf.consumers.remove(node)
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def squash(self, node, input_name):
        # remove the input buffer, causing that buffer's
        # producer to producer the output buffer instead.
        if input_name not in self.buffers:
            raise KeyError("Cannot squash node %s onto non-existent input buffer %s" % (node.op.name, input_name))
        input_buffer = self.buffers[input_name]
        output_buffer = self.buffers[node.output_names[0]]

        if len(input_buffer.consumers) > 1:
            raise ValueError("Cannot squash node %s onto input buffer %s, which has more than one consumer"
                             % (node.op.name, input_name))
        if node not in input_buffer.consumers:
            raise ValueError("Cannot squash node %s onto input buffer %s that it doesn't consume"
                             % (node.op.name, input_name))

        prev = input_buffer.producer
        output_idx = prev.output_names.index(input_name)
        prev.output_names[output_idx] = output_buffer.name
        output_buffer.producer = prev

        del self.buffers[input_name]
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def get_matched_nodes(self, sequence, validator=None):
        """
        Traverses each node in graph to find the requested pattern
        :param sequence: list[tuples] a list of node translation keys with their inputs and outputs. i.e:
                         each tuple contains ("opdapter.<op_name>.TRANSLATION_KEY", ([inputs]), ([outputs]))
                         The tuple for inputs/outputs should state BufferCriteria to verify list length; additionally,
                         each input/output should state specific BufferCriteria to determine how many(if any) of the
                         buffer should be in the matched sequence.
             E.g for format:
             sequence = [
                   # node type A
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...]),
                       # outputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
                   ),
                   # node type B
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (),
                       # outputs
                       ()
                   ),
                   ...
             ]
             E.g (Channel Shuffle). Note: we can pass strings instead of class.xxx for convenience, this function handles
                                          both.
             sequence = [
                        ("reshape",
                            (),
                            ("MATCH_NUM_BUFS", [("permute", "ALL")])
                        ),
                        ("permute",
                            (),
                            ("MATCH_NUM_BUFS", [("reshape", "ALL")])
                        ),
                        ("reshape",
                            (),
                            ()
                        )
                       ]
             Note 1: both inputs and outputs should also be translation keys
             Note 2: BufferCriteria can either be one of the BufferCriteria Enums or an INT to match a specific index
             Note 3: it is not required to have inputs or outputs, they can be left empty.
        :param validator: function to run if a match is found based on sequence. The matched sequence will be passed as
                          {"node_tuples": (nodes_matched)}
                          If not provided, function will return based on only matching the sequence as criteria.
        :return: list of node tuples that match the sequence provided, where each tuple contains the corresponding nodes
                 for each TRANSLATION_KEY in the sequence.
        """

        matched_nodes = []
        requested_types_seq = [entry[0].lower() for entry in sequence]
        start = 0
        end = len(sequence)
        nodes_list = self.list_nodes()

        log_debug2("Evaluating to match Sequence {}...", requested_types_seq)

        # we want to allow use of strings for op translation_keys(i.e op_types) to make sequence length minimal
        # so validate user has asked to match op_types that are supported in op_adapter
        log_assert(self.verify_op_types_exist(requested_types_seq) is True,
                   get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(requested_types_seq))

        while end <= len(nodes_list):
            nodes_tuple = tuple(nodes_list[start:end])  # get number of nodes based on length of sequence
            current_types_seq = [node.op.type for node in nodes_tuple]
            if (current_types_seq == requested_types_seq and self._validate_nodes_topology(nodes_tuple, sequence)) and \
                    (validator is None or validator(nodes_tuple)):
                matched_nodes.append(nodes_tuple)
                start = end  # start next node by skipping over the length of the sequence matched
                end += len(sequence)
            else:
                start += 1
                end = start + len(sequence)

        log_debug2("Found {} match(es)", len(matched_nodes))

        return matched_nodes

    def _validate_nodes_topology(self, nodes_tuple, sequence):
        """
        validates the input and output buffers for each matched node sequence in graph

        :param nodes_tuple: a tuple of matched nodes based on pattern
        :param sequence: the original list of sequences provided by user
        :return: True if each node's input and output buffer match the expected ones in sequence, False otherwise
        :raises: AssertionError if length and node types of node_list and sequence do not match
        """

        log_assert(len(nodes_tuple) == len(sequence), "Matched node list length must be same as requested sequence. "
                                                      "Expected {}, Got {}", len(nodes_tuple), len(sequence))

        for i in range(0, len(nodes_tuple)):
            node_type_actual = nodes_tuple[i].op.type
            node_type_expected = sequence[i][0]
            log_assert(node_type_actual == node_type_expected,
                       "Cannot validate topology for nodes of different types. Expected {}, Got{}",
                       node_type_expected, node_type_actual)

            inputs_actual = self.get_input_op_types(nodes_tuple[i])
            outputs_actual = self.get_output_op_types(nodes_tuple[i])
            inputs_expected, outputs_expected = sequence[i][1:]

            # providing inputs_expected and outputs_expected is not required from user
            # since user might just care to match a sequence of node types for any given inputs/outputs
            if (len(inputs_expected) and not self._validate_buffers(inputs_expected, inputs_actual)) or \
               (len(outputs_expected) and not self._validate_buffers(outputs_expected, outputs_actual)):
                    log_debug2("Sequence pattern {} matched, but not input/output buffers for node {} of type {} in "
                               "sequence.", [entry[0] for entry in sequence], nodes_tuple[i].op.name,
                               nodes_tuple[i].op.type)
                    return False

        return True

    def _validate_buffers(self, expected_buffers, actual_buffers):
        """
        validates the actual buffers(inputs or outputs of nodes) against the criteria set in the expected buffers
        :param expected_buffers: a tuple with BufferCriteria for matching the list of buffers, list of tuple pairs
                                 with each tuple containing the type of op and a buffer criteria
                        (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
        :param actual_buffers: list of actual buffer types for the current node being evaluated
        :return: true if actual buffers pass criteria set in the expected buffers, False otherwise

        raises Assertion error: if unknown buffer criteria,
               Value error: if ALL criteria given and there exists more expected inputs
        """

        # remove matching criteria from expected buffers and validate
        matching_criteria, expected_buffers = expected_buffers
        matching_criteria = matching_criteria.upper()
        log_assert(matching_criteria in [BufferCriteria.MATCH_NUM_BUFS, BufferCriteria.FLEXIBLE_NUM_BUFS],
                   get_error_message("ERROR_UNKNOWN_MATCHING_CRITERIA")
                   ([BufferCriteria.MATCH_NUM_BUFS, BufferCriteria.FLEXIBLE_NUM_BUFS], matching_criteria))

        if matching_criteria == BufferCriteria.MATCH_NUM_BUFS and len(expected_buffers) != len(actual_buffers):
            return False

        for op_type, buf_criteria in expected_buffers:
            op_type = op_type.lower()
            log_assert(self.verify_op_types_exist(op_type) is True,
                       get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(op_type))

            if type(buf_criteria) == int:
                if matching_criteria == BufferCriteria.MATCH_NUM_BUFS:
                    # User knows the number of input/output buffers to expect, hence it is an error to request
                    # an out-of-range index
                    log_assert(buf_criteria < len(actual_buffers), get_error_message("ERROR_BUFFER_CRITERIA_INDEX")
                               (op_type, buf_criteria, len(actual_buffers)))
                # In this case, user doesnt know/care for the number of input/output buffers of a node but want to
                # match ops that fit a certain criteria e.g. when the 2nd input is a particular op type;
                # in this instance an out-of-range index is not an error.

                if buf_criteria >= len(actual_buffers) or actual_buffers[buf_criteria] != op_type:
                    return False
            elif buf_criteria.upper() == BufferCriteria.ALL:
                if len(expected_buffers) != 1:
                    raise ValueError(get_error_message("ERROR_BUFFER_CRITERIA_ALL")
                                     (op_type, len(expected_buffers)))
                if not all(buf == op_type for buf in actual_buffers):
                    return False

            elif buf_criteria.upper() == BufferCriteria.ANY:
                if not any(buf == op_type for buf in actual_buffers):
                    return False

            elif buf_criteria.upper() == BufferCriteria.NONE:
                if any(buf == op_type for buf in actual_buffers):
                    return False

            # Unknown buffer criteria, so raise error
            else:
                raise ValueError(get_error_message("ERROR_UNKNOWN_BUFFER_CRITERIA")
                                 (op_type, ["ALL", "ANY", "NONE"], buf_criteria))

        return True

    @staticmethod
    def verify_op_types_exist(op_list):
        if type(op_list) is not list:
            op_list = [op_list]
        # get all supported op_types in op_adapter module
        supported_op_list = [class_[1].TRANSLATION_KEY if hasattr(class_[1], 'TRANSLATION_KEY') else ''
                             for class_ in inspect.getmembers(op_adapter, inspect.isclass)]
        return all(op_type in supported_op_list for op_type in op_list)

    def get_input_buffers(self, node):
        return [self.buffers[name] for name in node.input_names]

    def get_output_buffers(self, node):
        return [self.buffers[name] for name in node.output_names]

    def get_input_op_types(self, node):
        return [self.buffers[name].producer.op.type for name in node.input_names]

    def get_output_op_types(self, node):
        consumer_nodes = []
        consumer_nodes_types = []
        for name in node.output_names:
            for consumer in self.buffers[name].consumers:
                # consumer already existing in our list can happen if one consumer takes 2 or more outputs of a node.
                # e.g: if node_a has buf_1, buf_2 as outputs and next layer(node_b) has both of these buffers as input,
                # both buf_1 and buf_2 will list node_b as consumers so we don't want to have [node_b, node_b]
                # for outputs
                if consumer not in consumer_nodes:
                    consumer_nodes.append(consumer)
                    consumer_nodes_types.append(consumer.op.type)
        return consumer_nodes_types

    def get_buffer(self, buffer_name):
        return self.buffers[buffer_name]

    def has_buffer(self, buffer_name):
        return buffer_name in self.buffers

    def list_nodes(self):
        return self.nodes_in_order[:]

    def list_buffers(self):
        return list(self.buffers.values())
