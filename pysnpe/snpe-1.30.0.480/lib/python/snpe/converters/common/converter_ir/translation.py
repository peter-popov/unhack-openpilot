# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from snpe.converters.common.utils import code_to_message


class Translation(object):
    def __init__(self):
        self.indexed_methods = {}

    def apply_method(self, method_name, *args):
        return self.indexed_methods[method_name](*args)

    def register_method(self, method_name, method):
        self.indexed_methods[method_name] = method

    def has_indexed_method(self, method_name):
        return method_name in self.indexed_methods


class TranslationBank(object):
    def __init__(self):
        # string type name -> translation
        # the same value may exist for multiple keys.
        self.translations = {}

    def __get_translation(self, op_type):
        if op_type not in self.translations:
            raise KeyError("No translation registered for op type %s. "
                           "Op is most likely not supported by the converter." % op_type)
        return self.translations[op_type]

    def apply_method_to_op(self, op_type, method_name, *args):
        """
        Runs the requested method for the given op
        :param method_name: name of the method to call
        :param op_type: the operation type used to query the translation bank
        :param args: required positional arguments that will be passed to method

        raises KeyError if method not found for the requested op_type

        """
        translation = self.__get_translation(op_type)
        if not translation.has_indexed_method(method_name):
            raise KeyError("Translation for '%s' does not define an indexed method '%s'" % (op_type, method_name))
        return translation.apply_method(method_name, *args)

    def apply_method_to_all_ops(self, method_name, graph, *args, **kwargs):
        """
        Runs the requested method on all ops in the given graph. i.e loops through the nodes in graph, gets the
        corresponding translation class for node and runs the given method with node and graph as args
        :param method_name: name of the method to call
        :param graph: the IR Opgraph to traverse
        :param args: required positional arguments that will be passed to method
        :param kwargs: keywords arguments to be used to pass fail_if_no_method to this function

        raises KeyError if method no found for an op_type translation unless fail_if_no_method is set to False
                        in the kwargs argument
        """
        for node in graph.list_nodes():
            if node in graph.list_nodes():  # this extra check is needed since an optimization applied to op below
                                            # can remove next node(s), so doing a dynamic check is needed
                translation = self.__get_translation(node.op.type)
                if translation.has_indexed_method(method_name):
                    self.apply_method_to_op(node.op.type, method_name, node, graph, *args)
                else:
                    fail_if_no_method = kwargs.get('fail_if_no_method', True)
                    if fail_if_no_method:
                        raise KeyError(code_to_message.get_error_message("ERROR_METHOD_NOT_FOUND_FOR_OP_TYPE")
                                       (method_name, node.op.type))

    def apply_method_to_graph(self, method_name, graph, *args, **kwargs):
        """
        Runs the requested method on graph. i.e loops through the registered translation classes and runs the
        given method with the graph as argument
        :param method_name:  name of the method to call
        :param graph: the IR Opgraph to traverse
        :param args: required positional arguments that will be passed to method
        :param kwargs: keywords arguments to be used to pass fail_if_no_method to this function

        raises KeyError if method no found for an op_type translation unless fail_if_no_method is set to False
                        in the kwargs argument
        """

        for type_name, translation in self.translations.items():
            if translation.has_indexed_method(method_name):
                translation.apply_method(method_name, graph, *args)
            else:
                fail_if_no_method = kwargs.get('fail_if_no_method', True)
                if fail_if_no_method:
                    raise KeyError(code_to_message.get_error_message("ERROR_METHOD_NOT_FOUND_FOR_OP_TYPE")
                                   (method_name, type_name))

    def register_translation(self, translation, *op_types):
        for op_type in op_types:
            if op_type in self.translations:
                raise KeyError("A translation is already registered for op type '%s'" % op_type)
            self.translations[op_type] = translation

# Translation base class to be used across converters to translate source framework to IR IROpGraph
# method keys
ADD_INPUT_OP = "ADD_INPUT_OP"
ADD_OP = "ADD_OP"
INFER_SHAPE = "INFER_SHAPE"
POPULATE_AXES = "POPULATE_AXES"


class ConversionTranslationBase(Translation):
    def __init__(self):
        Translation.__init__(self)
        self.register_method(ADD_OP, self.add_op)
        self.register_method(INFER_SHAPE, self.infer_output_shapes)
        self.register_method(POPULATE_AXES, self.populate_axes_format)

    def add_op(self, src_op, graph):
        op = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        node = graph.add(op, input_names, output_names)
        self.populate_axes_format(node, graph)

    def extract_parameters(self, src_op, graph):
        raise NotImplementedError("extract_parameters for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, graph):
        raise NotImplementedError("extract_input_names for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_output_names(self, src_op, graph):
        raise NotImplementedError("extract_input_names for {} not implemented ".format(str(self.__class__.__name__)))

    def infer_output_shapes(self, op, input_shapes):
        return [input_shapes[0]]

    def populate_axes_format(self, node, graph):
        raise NotImplementedError("populate_axes_format for {} not implemented ".format(str(self.__class__.__name__)))