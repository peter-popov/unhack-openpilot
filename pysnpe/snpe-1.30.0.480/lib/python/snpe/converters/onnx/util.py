# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from functools import reduce
from operator import mul

try:
    import onnx
    from onnx import defs
    from onnx.numpy_helper import to_array as extract_onnx_tensor
except:
    onnx = None # converter will throw before we try anything in here

from snpe.converters.common.utils import code_to_message
from snpe.converters.common.utils.snpe_converter_utils import *


code_to_enum = {'i': onnx.AttributeProto.INT,
                'f': onnx.AttributeProto.FLOAT,
                's': onnx.AttributeProto.STRING,
                't': onnx.AttributeProto.TENSOR,
                'g': onnx.AttributeProto.GRAPH,
                'li': onnx.AttributeProto.INTS,
                'lf': onnx.AttributeProto.FLOATS,
                'ls': onnx.AttributeProto.STRINGS,
                'lt': onnx.AttributeProto.TENSORS,
                'lg': onnx.AttributeProto.GRAPHS}

KNOWN_ATTRIBUTE_DEFAULTS = dict(dilations=[1, 1],
                                strides=[1, 1],
                                pads=[0, 0, 0, 0],
                                output_shape=[],
                                axes=[],
                                consumed_inputs=[])


def parse_out_weights_biases_inputs(onnx_op, graph):
    """
    Checks if OP has constant weights or biases.
    :param onnx_op: onnx operation
    :param graph: the converter IR graph
    :return: tuple([const_input_names], [non_const_input_names])
    """
    input_names = list(map(str, onnx_op.input))
    weight_biases_inputs = []
    actual_inputs = []

    # check if any input OP has a bias or weights attribute(i.e Conv, BN...). This would lead to assume that the
    # rest input without those attributes are bias or weights themselves
    if any(hasattr(graph.get_buffer(name).producer.op, "bias") or hasattr(graph.get_buffer(name).producer.op, "weights")
           for name in input_names):
        for name in input_names:
            # to be considered a weight or bias input, it must be listed in initializer AND either the input buffer
            # must not be produced by an IR graph node or if in IR graph the node itself must not have a bias
            # or weights attribute(i.e most likely is a ConstantOp or StaticOp node)
            if graph.weights.has(name) and (not graph.has_buffer(name) or
                                            (not hasattr(graph.get_buffer(name).producer.op, "bias") and
                                             not hasattr(graph.get_buffer(name).producer.op, "weights"))):
                weight_biases_inputs.append(name)
            else:
                actual_inputs.append(name)
    else:
        actual_inputs = input_names

    return weight_biases_inputs, actual_inputs


def is_broadcast(onnx_op, graph=None):
    attrs = extract_attributes(onnx_op, [('axis', 'i', 0), ('broadcast', 'i', 0)])

    if graph is not None:
        # newer version of onnx(e.g version 7 of Mul or Add) do not have axis and broadcast attributes
        # hence another way to check would be to make sure all inputs to op are the same shape
        input_names = list(map(str, onnx_op.input))
        input_buffers_shape = []
        for name in input_names:
            if graph.has_buffer(name):
                input_buffers_shape.append(list(graph.get_buffer(name).shape))
            else:
                input_buffers_shape.append(list(graph.weights.fetch(name).shape))
        if any(shape != input_buffers_shape[0] for shape in input_buffers_shape):
            return True

    return attrs['axis'] != 0 or attrs['broadcast'] == 1


def assert_no_broadcast(onnx_op):
    log_assert(not is_broadcast(onnx_op),
               code_to_message.get_error_message("ERROR_BROADCAST_NOT_SUPPORTED")(onnx_op.name))


class NamedDict(dict):
    def __getattr__(self, key):
        return self[key]


def extract_attributes(onnx_op, attr_infos=None, schema=None, validate=False):
    """Ensure the existence and extract well typed attributes from an onnx
    NodeProto.
    :param attr_infos: a list of attributes to extract in the form [(attr_name, attr_type, attr_value)]
    :param schema:   an op_schema object for the onnx_op
    :param validate:  an optional validator function that is registered with the schema
                     of the form:  validator(src_op, attr_name, attr_value)

    Each entry in attr_info should be either a 2- or 3-tuple.
    * The first element should be the string name of an attribute.
    * The second element should by a type code for the attribute corresponding to:
      - i for int attributes
      - f for float attributes
      - s for string attributes
      - t for tensor attributes (returned as a numpy array)
      - g for graph attributes
      - lx, where x is one of the preceding attribute type identifiers, for list valued attributes
    * The third element, if present, specifies a default value should the attribute not be present.
      If no default is specified, this function will thrown an error.

    The return object will have a named property for each attribute info."""
    onnx_attrs = {}
    if not attr_infos and schema:
        attr_infos = schema.attributes()

    for attr in onnx_op.attribute:
        onnx_attrs[attr.name] = attr
        if schema and not validate:
            if not schema.check_unsupported_attributes(str(attr.name)):
                log_warning(code_to_message.get_warning_message("WARNING_UNSUPPORTED_ATTRIBUTE")
                            (attr.name, onnx_op.op_type, onnx_op.input[0]))

    ret = NamedDict()
    for attr_info in attr_infos:
        name = attr_info[0]
        if not name in onnx_attrs:
            if len(attr_info) == 3:
                ret[name] = attr_info[2]
                continue
            else:
                try:
                    ret[name] = KNOWN_ATTRIBUTE_DEFAULTS[name]
                    continue
                except KeyError:
                    raise ValueError(code_to_message.get_error_message("ERROR_ATTRIBUTE_MISSING")(onnx_op.name, name))
        attr = onnx_attrs[name]
        code = attr_info[1]
        requested_type = code_to_enum[code]
        if attr.type != requested_type:
            msg = code_to_message.get_error_message("ERROR_ATTRIBUTE_WRONG_TYPE")(onnx_op.name,
                                                                                  name,
                                                                                  onnx.AttributeProto.AttributeType.Name(requested_type),
                                                                                  onnx.AttributeProto.AttributeType.Name(attr.type))
            raise TypeError(msg)
        value = extract_onnx_type(code, attr)

        if validate and schema:
            schema.validate_data_constraints(onnx_op)
            schema.get_validate_method("validate_attribute_values")(onnx_op, name, value)
        ret[name] = value

    return ret


def extract_onnx_type(code, attr):
    ret = ''
    if code == 'i':
        ret = int(attr.i)
    elif code == 'f':
        ret = float(attr.f)
    elif code == 's':
        ret = str((attr.s).decode('utf-8'))
    elif code == 'g':
        ret = attr.g
    elif code == 't':
        ret = extract_onnx_tensor(attr.t)
    elif code == 'li':
        ret = list(map(int, attr.ints))
    elif code == 'lf':
        ret = list(map(float, attr.floats))
    elif code == 'ls':
        ret = list(map(str, attr.strings))
    elif code == 'lg':
        ret = list(attr.graphs)
    elif code == 'lt':
        ret = list(map(extract_onnx_tensor, attr.tensors))
    return ret


def extract_padding_mode(auto_pad, node_name):
    if auto_pad == 'VALID':
        return "PADDING_SIZE_IMPLICIT_VALID"
    elif auto_pad == 'SAME_LOWER':
        return "PADDING_SIZE_IMPLICIT_SAME"
    elif auto_pad == 'NOTSET':
        return "PADDING_SIZE_EXPLICIT_FLOOR"
    else:
        raise ValueError(code_to_message.get_error_message("ERROR_PADDING_TYPE_UNSUPPORTED")(node_name, auto_pad))


def set_to_weights_and_biases(src_op, graph, mode='bias'):
    """
     Sets a scale op's const input to either a weight or a bias depending on the mode (which can be either 'weights'
     or 'bias'). It assumes that if a scale op has const input then it can either be a weight or a bias if it follows
     an op that has a weight or bias. The exception is if the shapes of the const input and the prev op's weight/bias
     do not match or are not broadcastable to each other.

     The scale ops are:
        elementwise_sum
        elementwise_sub
        elementwise_product
        elementwise_div

    :param src_op: The onnx op
    :param graph:  The IRGraph object
    :param mode:   is either 'bias' or 'weights' depending on the nature of the scaling arithmetic.
    :return:       a tuple of the non-const inputs, non-zero biases and weights.
                   a tuple of non-const inputs, with one of the inputs broadcasted to match the previous op and
                   re-inserted into the graph, bias of NoneType and a weight of NoneType. The latter occurs when the op
                   shape does not match
    """
    weights = None
    bias = None
    const_input, actual_inputs = parse_out_weights_biases_inputs(src_op, graph)
    prev_ = graph.get_buffer(actual_inputs[0]).producer

    if len(const_input):
        log_assert(len(const_input) == 1, code_to_message.get_error_message("ERROR_MULTIPLE_CONST_INPUTS_FOUND")
                                                                           (src_op.op_type, const_input))

        const_value = graph.weights.fetch(const_input[0])
        if mode == 'bias':
            # if the previous op has a weight input or bias with the same shape, then we know the bias can be absorbed
            # without changing the final output.
            if hasattr(prev_.op, 'bias') and prev_.op.bias.shape != const_value.shape:
                pass
            else:
                bias = const_value
                return actual_inputs, weights, bias
        elif mode == 'weights':
            # if the previous op has a bias input or weights that are broadcastable with each other,
            # then we know the weights can be absorbed without changing the final output.
            if hasattr(prev_.op, 'weights') and not broadcastable(prev_.op.weights.shape, const_value.shape):
                pass
            else:
                weights = const_value
                return actual_inputs, weights, bias

        # if the code has not returned by this point, it means there is a shape mismatch between the inputs
        # which may be either a bias or a weight.
        # we take the const input and broadcast into according to the actual input (if possible),
        # and then pass it back into the graph.
        const_broadcast_value = broadcast_to(const_value, graph.get_buffer(actual_inputs[0]).shape)
        graph.weights.insert(const_input[0], const_broadcast_value)
        actual_inputs = [actual_inputs[0], const_input[0]]

    return actual_inputs, weights, bias


def broadcast_to(data, new_shape):
    """
    Broadcasts data into a new shape if possible
    :param new_shape: shape to be broadcasted into
    :param data: data to be broadcasted
    :return: broadcasted data if possible or original data if not
    """
    if data.shape != new_shape and broadcastable(data.shape, new_shape):
        return numpy.broadcast_to(data, new_shape).astype(numpy.float32)
    return data


def broadcastable(old_shape, new_shape):
    """
    Checks if two shapes are can be broadcast into one another in the numpy sense.
    :param old_shape: Shape of the old data
    :param new_shape: Desired broadcast shape
    :return: boolean if broadcast is possible otherwise false
    """
    if not isinstance(new_shape, list):
        new_shape = list(new_shape)
    while len(new_shape) != len(old_shape):
        new_shape.insert(0, 1)
    return all([(new_idx == 1 or new_idx == old_idx)
                for old_idx, new_idx in zip(old_shape, new_shape)])


def pads_symmetric(pads):
    num_dims = len(pads)//2
    for i in range(num_dims):
        if pads[i] != pads[i+num_dims]:
            return False
    return True


def pads_righthanded(pads):
    num_dims = len(pads)//2
    for i in range(num_dims):
        if pads[i] != 0:
            return False
    # don't call all zeros right-handed
    return not all(x == 0 for x in pads)


def product(nums):
    if len(nums) == 0:
        return 1
    else:
        return reduce(mul, nums)


class WeightData(object):
    def __init__(self, weights):
        # Weights from the network
        self.weights = weights
        # Track if the weights have been retrieved for use in another layer
        # Weights can be provided in one of two ways: initializers or constant ops
        # Constant ops being used as weight providers are setup with the weights from
        # the start and thus don't need to retrieve weights from the weight provider
        # again. SNPE layers like Conv/Matmul/GEMM/etc store weights internally and
        # will attempt to retrieve the weights. The consumed field will track which
        # Constant ops are being used as weight providers so they can be pruned from
        # the network at the end
        self.consumed = False


# ------------------------------------------------------------------------------
#   WeightProvider
# ------------------------------------------------------------------------------
class WeightProvider(object):
    def __init__(self, model):
        self.weight_map = {}
        for tensor in model.graph.initializer:
            self.weight_map[str(tensor.name)] = WeightData(extract_onnx_tensor(tensor))

    def consumed(self, key):
        if not key in self.weight_map:
            return False
        return self.weight_map[key].consumed

    def fetch(self, *keys, **kwargs):
        ret = []
        # Prunable indicates whether the weights have been consumed in such a way as to
        # allow pruning of the node (eg Const ops that contain weights are consumed by
        # Conv/FC/etc and thus can be pruned from the network. Const ops that are inputs
        # to a node cannot
        consumed = kwargs.get('prunable', True)
        for key in keys:
            key = str(key)
            log_debug(code_to_message.get_debugging_message("DEBUG_RETRIEVE_WEIGHTS, key"))
            if key not in self.weight_map:
                raise KeyError(code_to_message.get_error_message("ERROR_WEIGHTS_MISSING_KEY")(key))
            self.weight_map[key].consumed = consumed
            # Explicitly copy the data so if later ops modify it, the original data remains intact
            ret.append(numpy.require(self.weight_map[key].weights.copy(), dtype=numpy.float32))
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def has(self, key):
        return key in self.weight_map

    def has_all(self, keys):
        return all(self.has(key) for key in keys)

    def insert(self, key, weights):
        log_debug("Inserting weights for {}", key)
        self.weight_map[key] = WeightData(weights)

