# -*- mode: python -*-
import argparse

try:
  from snpe.converters.onnx.op_schema import *
  from snpe.converters.onnx.util import *
except ImportError:
    raise ImportError('Please include SNPE library tools in your python path')

import onnx
import sys
import os

LATEST_SUPPORTED_ONNX_VERSION = 7 # current maximum supported version for any op in the converter


# ------------------------------------------------------------------------------
#   Helper functions
# ------------------------------------------------------------------------------
def col_width(layer_element):
    new_col_width = 0
    for col in layer_element:
        if col:
            new_col_width = max(new_col_width, sum(map(len, col)))
    return new_col_width


def format_arg(element):
    if len(element) == 2:
        return "{}:{}".format(element[0], element[1])
    elif len(element) == 1:
        return "{}".format(element[0])
    else:
        return ""


def printable(layer):
    for i in range(1, len(layer)):
        for j in range(0, len(layer[i])):
            if len(layer[i][j]) == 2:
                return True
    return False


def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.
    """
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or
                                                  'ANSICON' in os.environ)
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True


def prGreen(skk):
    """
    Returns string in green color.
    """
    colored_skk ="\033[92m{}\033[00m".format(skk)
    if supports_color():
        print(colored_skk)
        return
    print(skk)


def prRed(skk):
    """
    Returns string in red color.
    """
    colored_skk = "\033[91m{}\033[00m" .format(skk)
    if supports_color():
        log_warning(colored_skk)
        return
    log_warning(skk)


def prYellow(skk):
    """
    Returns string in yellow color.
    """
    colored_skk = "\033[93m{}\033[00m".format(skk)
    if supports_color():
        print(colored_skk)
        return
    print(skk)


def prBlue(skk):
    """
    Returns string in red color.
    """
    colored_skk = "\033[94m{}\033[00m".format(skk)
    if supports_color():
        print (colored_skk)
        return
    print(skk)


# ------------------------------------------------------------------------------
#   Validators, Checkers
# ------------------------------------------------------------------------------
def validate_data_constraints(schema, src_op, debug=False):
    """
    Checks to see if the number of inputs and outputs in the src_op match the schema
    :param schema:  The op_schema object
    :param src_op: The onnx op
    :param debug:  displays outputs messages if True (debug mode), returns a list if false (info mode)
    :return: a list if debug is set to false
    """
    max_input_length = schema.numInputs
    max_output_length = schema.numOutputs
    unused_input_data = []
    unused_output_data = []

    # first check number of inputs and outputs match
    if max_input_length and len(src_op.input) > max_input_length:
        if debug:
            prYellow("Expected {} input(s) but found {}. One of the following inputs: {} from {} op "
                     "will not be used".
                     format(max_input_length, len(src_op.input), list(map(str, src_op.input[:])), src_op.op_type))
        else:
            extra = ['unused'] * (len(src_op.input) - max_input_length)
            unused_input_data = list(zip(src_op.input[max_input_length:], extra))

    if max_output_length and len(src_op.output) > max_output_length:
        if debug:
            prYellow("Expected {} input(s) but found {}. One of the following inputs: {} from {} op will "
                     "not be used".
                     format(max_output_length, len(src_op.output), src_op.outputs[:], src_op.op_type))
        else:
            extra = ['unused'] * (len(src_op.output) - max_output_length)
            unused_output_data = list(zip(src_op.output[max_output_length:], extra))

    return unused_input_data, unused_output_data


def check_attribute_supported(src_op, op_version=LATEST_SUPPORTED_ONNX_VERSION, debug=True):
    """
    This function checks if an attribute is supported in the following order: if the attribute is known->
    if the attribute is of the right type-> if the attribute has an accepted value (values may be constrained by the
    converter)
    :param src_op: The Onnx op
    :param op_version: The opset version for src_op
    :param debug: If the function should display output messages (debug mode) or return a list (info mode)
    :return: a list if debug is set to false
    """
    op_name = src_op.op_type
    src_op_schema = get_op_schema(op_name, op_version)
    unsupported_attrs = []

    for attr in src_op.attribute:
        checked = src_op_schema.check_unsupported_attributes(attr.name)
        # check if attributes are present in the Onnx Model Op that are not supported by SNPE Onnx IR
        if not checked:
            if debug:
                prGreen(code_to_message.get_warning_message("WARNING_UNSUPPORTED_ATTRIBUTE")
                            (attr.name, src_op.op_type, src_op.input[0]))
            else:
                unsupported_attrs.append([attr.name, 'unsupported'])

        # if attribute is supported, then check type
        elif checked:
            attribute = src_op_schema.attributes(attr.name)
            if not check_attribute_type(attribute, attr):
                if debug:
                    prGreen(code_to_message.get_error_message("ERROR_ATTRIBUTE_WRONG_TYPE")
                                (src_op.op_type, attr.name, attr.type, code_to_enum[attribute[1]]))
                else:
                    unsupported_attrs.append([attr.name, 'wrong type'])
                continue

            # if type is good then check value
            attribute = src_op_schema.attributes(attr.name)
            try:
                extract_attributes(src_op, attr_infos=[attribute], schema=src_op_schema, validate=True)

            # this means that an attribute value is not supported
            except ValueError as v:
                if debug:
                    prGreen(v)
                elif not debug:
                    if attr.name not in dict(unsupported_attrs):
                        unsupported_attrs.append([attr.name, 'unsupported value'])

            # this means that translation failed to pass some other constraint
            except AssertionError as e:
                if debug:
                    prGreen(e)
                elif not debug:
                    if attr.name not in dict(unsupported_attrs):
                        unsupported_attrs.append([attr.name, 'invalid value'])

            except Exception as e:
                if debug:
                    print(e)
                else:
                    pass

    return unsupported_attrs


def check_op_supported(src_op, model_version=LATEST_SUPPORTED_ONNX_VERSION):
    try:
        schema_dict = get_op_schema(src_op.op_type, 0)
    except IndexError:
        return [[str(src_op.op_type), 'unsupported']]

    supported_version = [version for version in schema_dict.get_schemas()]
    op_version = onnx.defs.get_schema(src_op.op_type, model_version, '').since_version
    if str(op_version) not in supported_version:
        return [[str(src_op.op_type), 'unsupported version']]
    else:
        return [[str(src_op.op_type)]]


# ------------------------------------------------------------------------------
#   Dry_run functions
# ------------------------------------------------------------------------------

def setup_dry_run(model, mode='info'):
    layers = []
    name_length = len("Layers")
    op_type_length = len("Ops")
    input_length = len("Inputs")
    output_length = len("Outputs")
    args_length = len("Attributes")
    printable_layers = []
    model_version = [opset.version for opset in model.opset_import]

    if mode == 'debug':
        setup_dry_run_messages(model, model_version[-1])
        return

    for i, src_op in enumerate(model.graph.node):
        op_version = onnx.defs.get_schema(src_op.op_type, model_version[-1], '').since_version
        op_supported = check_op_supported(src_op, model_version[-1])

        # if the op is not supported, i.e not in the registry, we can't do anything else
        # so fill the checks with blanks
        # else we check attributes, inputs and outputs
        if len(op_supported[0]) == 2 and 'unsupported' == op_supported[0][1]:
            layer = ([[src_op.name if src_op.name else str(i)],
                      op_supported,
                      [],
                      [],
                      []])
        else:
            # if op version is not supported, we get the closest op schema
            # then we set the op version that the attributes are checked against
            # using that op_schema version
            # otherwise, if the op is supported we get the exact op_schema for that version
            if len(op_supported[0]) == 2:
                layer_op_schema = get_op_schema(src_op.op_type)
                op_version = layer_op_schema.version[0]
            else:
                layer_op_schema = get_op_schema(src_op.op_type, op_version)

            checked_data_constraints = validate_data_constraints(layer_op_schema, src_op)
            layer = ([[src_op.name if src_op.name else str(i)],
                      op_supported,
                      checked_data_constraints[0],
                      checked_data_constraints[1],
                      check_attribute_supported(src_op, op_version, debug=False)])

        # reformat layer, make all columns the same width and rows the same height
        max_list_len = max([len(layer[i]) for i in range(0, len(layer))])
        for j in range(0, len(layer)):
            while len(layer[j]) < max_list_len:
                layer[j].append('')

        name_length = max(name_length, len(layer[0][0]))
        op_type_length = max(op_type_length, col_width(layer[1]))
        input_length = max(input_length, col_width(layer[2]))
        output_length = max(output_length, col_width(layer[3]))
        args_length = max(args_length, col_width(layer[4]))
        layers.append(layer)

        # checks if there is anything to print
        if printable(layer):
            printable_layers.append(True)
        else:
            printable_layers.append(False)

    if supports_color():
        row_format = '\x1b[34m|{:<%d}| \x1b[32m {:<%d} | \x1b[31m {:<%d} | {:<%d} | {:<%d} | \x1b[0m' % (
            name_length, op_type_length + 1, input_length + 1, output_length + 1, args_length + 1)
    else:
        row_format = '|{:<%d}| {:<%d} | {:<%d} | {:<%d} | {:<%d} |' % (
            name_length, op_type_length + 1, input_length + 1, output_length + 1, args_length + 1)

    # check if there is anything to print
    if any(printable_layers):
        print(row_format.format("Layers", "Ops", "Inputs",
                                "Outputs", "Attributes"))
        for k, layer in enumerate(layers):
            for i in range(0, len(layer[4])):
                if printable_layers[k]:
                    print(row_format.format(layer[0][i], format_arg(layer[1][i]), format_arg(layer[2][i]),
                                            format_arg(layer[3][i]), format_arg(layer[4][i])))
    else:
        print("Model ops, op attributes, inputs and outputs have been evaluated")


def setup_dry_run_messages(model, model_version):
    for i, src_op in enumerate(model.graph.node):
        log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, src_op.op_type))
        op_supported = check_op_supported(src_op, model_version)
        op_version = onnx.defs.get_schema(src_op.op_type, model_version, '').since_version
        supported_version = []

        # check to see if an op is supported, we need the try in case it fails because the op
        # does not exist in the registry
        # otherwise
        if len(op_supported[0]) == 2:
            try:
                supported_version = map(int, get_op_schema(src_op.op_type, 0)._schemas.keys())
            except IndexError:
                prRed(code_to_message.get_warning_message("WARNING_OP_NOT_SUPPORTED")
                      (src_op.op_type))
                continue

            prRed(code_to_message.get_warning_message("WARNING_OP_VERSION_NOT_SUPPORTED")
                  (src_op.op_type, supported_version, [int(op_version)]))
            continue

        src_op_schema = get_op_schema(src_op.op_type, op_version)
        check_attribute_supported(src_op, op_version, debug=True)
        validate_data_constraints(src_op_schema, src_op, debug=True)
    log_info("Model ops, op attributes, inputs and outputs have been evaluated")


# ------------------------------------------------------------------------------
#  Info Functions
# ------------------------------------------------------------------------------
def print_attribute_info():
    attributes_supported = dict()

    for op_name in OP_SCHEMA_REGISTRY:
        schema = get_op_schema(op_name)
        attributes_supported[op_name] = [attribute[0] for attribute in schema.attributes()]

    max_key_length = max(map(len, attributes_supported.keys()))
    max_value_length = max(map(len, attributes_supported.values()))

    row_format = '{:<%d} | {:<%d}' % (max_key_length, max_value_length)

    prYellow('Printing supported attribute information for Onnx Converter Ops................')
    prYellow(row_format.format("Operator name", "Supported Attributes"))
    print ('--' * (max_key_length + max_value_length))
    for key, value in attributes_supported.items():
        prYellow(row_format.format(str(key), list(map(str, value))))


def print_version_info():
    version_supported = dict()
    for op_name, schema_dict in OP_SCHEMA_REGISTRY.items():
        version_supported[op_name] = list(schema_dict.get_schemas().keys())

    max_key_length = max(map(len, version_supported.keys()))
    max_value_length = max(map(len, version_supported.values()))

    row_format = "{:<%d} | {:<%d}" % (max_key_length, max_value_length)

    prBlue('Printing version information for supported Onnx Ops ................................')
    prBlue(row_format.format("Operator name", "Supported Onnx Versions"))
    prBlue('--' * (max_key_length + max_value_length))
    for key, value in version_supported.items():
        prBlue( row_format.format(key, value))


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", '-m')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--info', action='store_true', help='Prints information about the Onnx converter')
    args = parser.parse_args()

    if args.network:
        with open(args.network, 'rb') as f:
            model = onnx.load(f)
            if args.debug:
               setup_dry_run(model, mode='debug')
            else:
               setup_dry_run(model)
            log_info('Model ops, op attributes, inputs and outputs have been evaluated.')

    if args.info:
        print_version_info()
        print('\n')
        print_attribute_info()


