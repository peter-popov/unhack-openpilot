# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from collections import OrderedDict
try:
    import onnx
    from onnx.defs import *
except:
    onnx = None  # converter will throw before we try anything in here

from snpe.converters.common.utils import code_to_message
from snpe.converters.onnx.util import code_to_enum, extract_onnx_type
import copy

OP_SCHEMA_REGISTRY = dict() # a dictionary containing op_schema_dict objects for all supported ops.


# -----------------------------------------------------------------------------
#   Onnx Op Schema Base Class Definitions
# ------------------------------------------------------------------------------
class OpSchemaDict:

    """ This is a dictionary style class of op_schema objects tied to their corresponding
        versions for the same op_type. Useful for performing the same operation on all schemas registered to
        a single op.

        Usage:        E.x Mul_op_schema_dict = OpSchemaDict(op_name='Mul')
                          Mul_op_schema = OpSchemaBase(op_name='Mul')
                          Mul_op_schema_dict.add_schema(Mul_op_schema, version=1) """

    def __init__(self, op_name=''):
        self.op_name = op_name
        self._schemas = OrderedDict()

    def add_schema(self, schema, version):
        if isinstance(schema, OpSchemaBase):
            self._schemas[str(version)] = schema
        else:
            raise ValueError('Expected instance of {}, instead got {}'.format(OpSchemaBase, schema))

    def get_schemas(self, version=None):
        if version:
            return self._schemas[str(version)]
        return self._schemas

    def register_method(self, method):
        for schema in self._schemas.values():
            schema.register_method(method)

    def replace_default_values(self, **kargs):
        for schema in self._schemas.values():
            schema.replace_default_values(**kargs)

    def reorder_op_schemas(self, model_op_version):
        """
        This function reorders the op_schema_dict so that the last schema is always relevant to
        the model_op_version.
        :param model_op_version: The global model opset version
        """
        model_op_schema = OpSchemaBase()
        reordered_schema_dict = OrderedDict()

        while self._schemas:
            version, schema = self._schemas.popitem()
            if version == model_op_version:
                model_op_schema = schema
                continue
            reordered_schema_dict[version] = schema
        self._schemas.update(reordered_schema_dict)
        self._schemas[model_op_version] = model_op_schema


class OpSchemaBase:

    """ This is a wrapper class based on the Onnx spec schema, which defines an operator in the following form
        Op_schema:

          Attributes:
            op_name: the name of the op, this is the onnx src_op op_type.
            numInputs : The maximum number of inputs for this op
            numOutputs : The maximum number of outputs for this op
            version : The Onnx opset version for this op
            _attributes: the default attributes for this op
            methods: optional methods that validate constraints against either the inputs, outputs or the attributes.

          Usage:
            Create an op_schema:
                conv_op_schema= OpSchemaBase(op_name='Conv')
                onnx_op_schema = onnx.defs.C.get_schema('Conv', version, '')
                conv_op_schema.populate_op_schema(onnx_op_schema)

            Change the default values for an op_schema from onnx's default:
                onnx_default_pads = [0, 0, 0, 0]
                new_pads = [2, 2, 2, 2]
                conv_op_schema.replace_default_values(pads=new_pads)

            Add a new validator to check some constraint
             new_op_schema.register_method(my_validate_attribute_values)
             new_op_schema.register_method(my_validate_data_constraints)"""

    def __init__(self, op_name=''):
        self.op_name = op_name
        self.numInputs = None   # list of inputs, NoneType indicates no required number of inputs
        self.numOutputs = None  # list of outputs, NoneType indicates no required number of outputs
        self.version = []     # list of strings of supported versions
        self._attributes = {}  # list of attributes in the form "attribute_name: (name, type, value)"
        self.methods = {}

    def register_method(self, method):
        method_name = method.__name__
        method_copy = copy.deepcopy(method)
        self.methods[method_name] = method_copy

    def populate_op_schema(self, onnx_op_schema, *args):
        """
         Wraps an onnx schema into a condensed format
         - where attributes can be excluded if they are not supported in the converter
         - op_schema objects are tied to their respective version
         - attributes are assigned default values if not required

        :param onnx_op_schema: Takes in an onnx schema definition
        :param args: optional list of unsupported attributes to be parsed out of the onnx schema
        """
        self.op_name = onnx_op_schema.name
        self.version = [onnx_op_schema.since_version]
        self.numInputs = onnx_op_schema.max_input
        self.numOutputs = onnx_op_schema.max_output

        if args:
            unsupported_attrs = args[0]
        else:
            unsupported_attrs = []

        for attribute_name, attribute in onnx_op_schema.attributes.items():

            if attribute.name not in unsupported_attrs:
                self._attributes[str(attribute_name)] = parse_attributes(attribute)

            elif attribute.name in unsupported_attrs and attribute.required:
                raise AttributeError("Cannot exclude this attribute {} from this version {} since it is required".
                                     format(attribute.name, self.version))

    def validate_data_constraints(self, src_op):
        """
        Validates that the provided inputs/outputs have the same length. Intended to be overridden with custom
        constraints since the base function only performs a basic check.

        :param src_op:  Takes in a src_op and verifies any constraints on the number of inputs or outputs in the
                       schema
        :raises an IndexError if the len of the src_op inputs or src_op_outputs does not match the schema
        """
        if len(src_op.input) > self.numInputs:
            raise IndexError("Expected {} input(s) but found {}. One of the following inputs: {} from {} "
                             "op is invalid".
                             format(self.numInputs, len(src_op.input), list(map(str, src_op.input[:])), src_op.op_type))

        if len(src_op.output) > self.numOutputs:
            raise IndexError("Expected {} output(s) but found {}. One of the following outputs: {} from {} "
                             "op is invalid".
                             format(self.numOutputs, len(src_op.output), list(map(str, src_op.output[:])), src_op.op_type))

    @staticmethod
    def validate_attribute_values(src_op, attr_name='', attr_value=''):
        """
        Some Ops constrain attributes to values that can be processed by SNPE runtimes.
        This function checks if the src_op attribute values for the constrained attribute are different
        from the supported attribute default values.

        :param src_op: the raw Onnx op
        :param attr_name: the src_op attribute name
        :param attr_value: the src_op attribute value
        :return: nominally raises a value error if the attribute value is not supported.

        """
        value = attr_value
        schema = get_op_schema(src_op.op_type)
        default_attribute = schema.attributes(attr_name)

        # default_attribute are expected in the form (attr_name, attr_type) if required
        # and (attr_name, attr_type, attr_value) if optional. This conditional ensures that required
        # attributes are not checked against default values.
        if len(default_attribute) < 3 or not attr_name:
            return

        elif default_attribute[2] != value:
                raise ValueError(code_to_message.get_error_message("ERROR_UNSUPPORTED_ATTRIBUTE_VALUE"). \
                    format(attr_name, src_op.op_type, src_op.input[0],
                           default_attribute[2],
                           value))

    def attributes(self, names=None):
        if names:
            if isinstance(names, list):
                return [self._attributes[name] for name in names]
            else:
                return self._attributes[names]
        return list(self._attributes.values())

    def replace_default_values(self, **kargs):
        if kargs:
            for key, value in kargs.items():
                self._attributes[key] = [self._attributes[key][0], self._attributes[key][1], value]

    def get_validate_method(self, method_name):
        if method_name in self.methods:
            return self.methods[method_name]
        return NotImplementedError

    def check_unsupported_attributes(self, attribute=None):
        """
           Checks if an attribute is supported by the op_schema, or indirectly by the Onnx Converter.
           :param attribute: a list of attribute name(s) extracted from the onnx op
           :return: False if any of the provided attributes are not present in the op_schema for the specified op_version,
                    True otherwise

             Usage:
                   conv_op_schema = get_op_schema('Conv')
                   conv_op_schema.check_unsupported_attribute(['dilations']) returns True
                   conv_op_schema.check_unsupported_attribute(['epsilon']) returns False
           """
        snpe_attrs_names = []

        # check if attributes are present in the Onnx Model Op that are not supported by SNPE Onnx IR
        if isinstance(attribute, list):
            for attr in attribute:
                snpe_attrs_names = any([snpe_attrs[0] == attr for snpe_attrs in self.attributes()])
        else:
            snpe_attrs_names = [snpe_attrs[0] == attribute for snpe_attrs in self.attributes()]

        return any(snpe_attrs_names)


# ------------------------------------------------------------------------------
#   Onnx Op Schema Module Functions
# ------------------------------------------------------------------------------
def check_attribute_type(attribute, src_op_attr):
    snpe_type = code_to_enum[str(attribute[1])]
    src_type = src_op_attr.type
    return src_type == snpe_type


def get_op_schema(op_name, version=None):
    """
    Retrieves an op_schema from the registry.
    :param op_name: the name of the Onnx Op
    :param version: the version of the op_schema to return. Note: op_schemas are tied to an op version
    :return: A specific op_schema for the specified op_name if the version > 0
             All op_schemas for the specified op_name if version = 0
             The last op_schema for that op, which is determined according the global opset_version as
             per OpVersionInfo if no version is provided
    """
    try:
        op_schema_dict = OP_SCHEMA_REGISTRY[op_name.lower()]
        if version > 0:
            return op_schema_dict.get_schemas(str(version))
        elif version == 0:
            return op_schema_dict
        else:
            return list(op_schema_dict.get_schemas().values())[-1]
    except Exception as e:
        raise IndexError("{}: No schema found for this op: {}. Op is most likely not supported by the Converter"
                         "".format(e, op_name))


def parse_attributes(attribute):
    attributes = list()
    attributes.append(str(attribute.name))
    for key, value in code_to_enum.items():
        if attribute.type == value:
            attr_type = key
            attributes.append(attr_type)
            continue

    if attribute._default_value:
        ret = extract_onnx_type(attr_type, attribute.default_value)
        attributes.append(ret)

    return attributes

