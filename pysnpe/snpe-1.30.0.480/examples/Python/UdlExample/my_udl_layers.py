#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from snpe.converters.common.converter_ir.axis_tracker import AxisTracker
from snpe.converters.common.utils import snpe_udl_utils

# List your udl modules to import
import caffe_custom_scale_udl


def udl_factory_func():
    """
    Factory function that will be passed to caffe converter command line along with the name of this python module

    :return: a dictionary of {layer_type(type:str): udl object{type:Udl class)}
    """

    # Instance of Udl class for mycustomscale layer
    udl_mycustomscale = snpe_udl_utils.Udl(layer_callback=caffe_custom_scale_udl.udl_mycustomscale_func,
                                           expected_axes_orders=[
                                            # First supported input/output axes order (4D: NSC input, NSC output)
                                            (  # input dims
                                             [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                                              AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL],
                                             # output dims
                                             [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                                              AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
                                            ),
                                            # Second supported input/output axes order (3D: NS input, NS output)
                                            (  # input dims
                                             [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                                              AxisTracker.AxisAnnotations.WIDTH],
                                             # output dims
                                             [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                                              AxisTracker.AxisAnnotations.WIDTH]
                                            ),
                                            # Third supported input/output axes order (2D: NF input, NF output)
                                            (  # input_dims
                                             [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE],
                                             # output_dims
                                             [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE]
                                            )
                                           ])
    """
    Optionally, Users can set the above axes order as below
    """
    # # Add udl's expected input axis order for 4D: NSC input, NSC output
    # udl_mycustomscale.add_expected_axis_order(  # input dims
    #                                           [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
    #                                           AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL],
    #                                           # output dims
    #                                           [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
    #                                           AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL])
    #
    # # Add udl's expected input axis order for 3D: NS input, NS output
    # udl_mycustomscale.add_expected_axis_order(  # input dims
    #                                           [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
    #                                            AxisTracker.AxisAnnotations.WIDTH],
    #                                           # output dims
    #                                           [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
    #                                            AxisTracker.AxisAnnotations.WIDTH])
    #
    # # Add udl's expected input axis order for 2D: NF input, NF output
    # udl_mycustomscale.add_expected_axis_order(  # input_dims
    #                                          [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL],
    #                                           # output_dims
    #                                          [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL])

    # UDL layer name to UDL class map
    udl_supported_types = {
        'MyCustomScale': udl_mycustomscale
    }

    return udl_supported_types
