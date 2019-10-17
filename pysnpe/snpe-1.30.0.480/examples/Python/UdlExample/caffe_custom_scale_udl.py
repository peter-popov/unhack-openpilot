#!/usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import struct

from snpe.converters.common.utils.snpe_converter_utils import SNPEUtils
from snpe.converters.common.utils import snpe_udl_utils

snpeUtils = SNPEUtils()


class LayerType:
    MY_CUSTOM_SCALE_LAYER = 1
    MY_ANOTHER_LAYER = 2


class UdlBlobMyCustomScale(snpe_udl_utils.UdlBlob):
    """
    Wrapper class for MyCustomScale layer blob
    """
    def __init__(self, layer, weight_provider):
        snpe_udl_utils.UdlBlob.__init__(self)

        # MyCustomScale layer reuses the Caffe Scale layer params
        caffe_params = layer.scale_param

        # Initialize the SNPE params
        snpe_params = UdlBlobMyCustomScale.MyCustomScaleLayerParam()

        # fill the params
        snpe_params.bias_term = caffe_params.bias_term

        # fill the weights
        caffe_weights = snpeUtils.blob2arr(weight_provider.weights_map[layer.name][0])
        snpe_params.weights_dim = list(caffe_weights.shape)
        snpe_params.weights_data = list(caffe_weights.astype(float).flat)

        self._blob = snpe_params.serialize()
        self._size = len(self._blob)

    def get_blob(self):
        return self._blob

    def get_size(self):
        return self._size

    class MyCustomScaleLayerParam:
        """
        Helper class for packing blob data
        """
        def __init__(self):
            self.type = LayerType.MY_CUSTOM_SCALE_LAYER
            self.bias_term = None
            self.weights_dim = []
            self.weights_data = []

        def serialize(self):
            packed = struct.pack('i', self.type)
            packed += struct.pack('?', self.bias_term)
            packed += struct.pack('I%sI' % len(self.weights_dim),
                                  len(self.weights_dim), *self.weights_dim)
            packed += struct.pack('I%sf' % len(self.weights_data),
                                  len(self.weights_data), *self.weights_data)
            return packed


def udl_mycustomscale_func(layer, weight_provider, input_dims):
    """
    Conversion callback function for MyCustomScale layer
    """
    # Initialize blob for our custom layer with the wrapper class
    blob = UdlBlobMyCustomScale(layer, weight_provider)

    # Input and output dims are the same for MyCustomScale layer
    return snpe_udl_utils.UdlBlobOutput(blob=blob, out_dims=input_dims)
