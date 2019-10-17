#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


class Udl(object):
    def __init__(self, layer_callback, expected_axes_orders=None):
        """
        An object of this class should be the value(s) of the dictionary returned by the factory function passed to
        converter when using the --udl option. i.e {layer_type: Udl Object}

        :param layer_callback: python function instance. The converter will call this function to pass the layer object
                               and input dims. This function is expected to pack all of the necessary information into
                               a single blob along with the output dimensions. It should return an object of type
                               UdlBlobOutput class defined below which will be used to retrieve data by the Converter.
        :param expected_axes_orders: list of tuples containing supported axes order if different from default or if
                                     layer supports multiple axes orders(see below add_expected_axis_order(...) for
                                     default).
                                     Each tuple should contain (expected_input_axes_order, expected_output_axes_order)
                                     in that order. Optionally,the add_expected_axis_order(...)function can be used
                                     to populate axes_order instead.
        """
        self._layer_callback = layer_callback
        self._expected_input_axes_order = []
        self._expected_output_axes_order = []
        if len(expected_axes_orders):
            for order in expected_axes_orders:
                if len(order) != 2:
                    raise ValueError("Must provide 2 axes orders, one for input and another for output. Got {}",
                                     len(order))
                input_axes_order, output_axes_order = order
                assert(isinstance(input_axes_order, list))
                assert(isinstance(output_axes_order, list))
                self._expected_input_axes_order.append(input_axes_order)
                self._expected_output_axes_order.append(output_axes_order)

    def get_layer_callback(self):
        return self._layer_callback

    def add_expected_axis_order(self, input_axes_order, output_axes_order):
        """
        Optionally, this function could be used to specify UDL's expected input and output axes order
        during runtime. (i.e what is the custom runtime implementation's expected input and output axes order(s))

        By default, UDL's expected input/output axes order is of NSC (Batch, Spatial, Channel), i.e:
        { AxisAnnotation.Batch, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH, AxisAnnotation.CHANNEL }.
        Please use AxisTracker.AxisAnnotations from snpe.converters.common.converter_ir.axis_tracker to add
        your axes orders

        If UDL can handle multi-dimensional inputs i.e. 3D, 2D and 1D,this function needs to be called for each
        input rank or must be listed in the above constructor.
        """
        assert(isinstance(input_axes_order, list))
        assert(isinstance(output_axes_order, list))
        self._expected_input_axes_order.append(input_axes_order)
        self._expected_output_axes_order.append(output_axes_order)

    def get_expected_axis_order(self):
        return self._expected_input_axes_order, self._expected_output_axes_order


class UdlBlob(object):
    """The User Defined Class containing blob data and size
    A class object that will be passed to below UdlBlobOutput class must at a minimum implement the following two
    functions which will be called by converter to retrieve the packed blob(along with the size) to be added to IR
    graph. Users can use this class as base class and overwrite the functions(as shown in example)
    """
    def __init__(self):
        self._blob = ''
        self._size = len(self._blob)

    def get_blob(self):
        return self._blob

    def get_size(self):
        """should return size of self.blob"""
        return self._size


class UdlBlobOutput(object):
    """
    Class that will be used to hold an object of type UdlBlob class(or similarly defined by user) plus the expected
    output dimensions.
    *** Users are expected to return an Instance/Object of this class from their callback function.
    """
    def __init__(self, blob, out_dims):
        self._blob = blob
        # _out_dims is a list of lists
        # a list where each index is a list of the dims
        self._out_dims = out_dims
        assert(isinstance(self._out_dims, list))
        for dims in self._out_dims:
            assert(isinstance(dims, list))

    def get_blob(self):
        return self._blob

    def get_output_dims(self, idx):
        return self._out_dims[idx]
