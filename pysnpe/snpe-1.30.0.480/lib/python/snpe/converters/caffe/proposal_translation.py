# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import yaml

from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from snpe.converters.common.converter_ir import op_adapter
from snpe.converters.common.utils import code_to_message
from snpe.converters.common.utils.snpe_converter_utils import *


# -----------------------------------------------------------------
# Converter translations
# -----------------------------------------------------------------
class CaffeProposalTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        python_param = layer.python_param
        py_layer = python_param.layer
        py_param_str = python_param.param_str

        if py_layer != 'ProposalLayer':
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_UNSUPPORTED_PYTHON_MODULE')
                             (str(layer.name), py_layer))
        if not len(py_param_str):
            raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_PROPOSAL_LAYER_MISSING_PARAM_STR_FIELD')
                             (str(layer.name)))

        layer_params = yaml.load(py_param_str)
        feat_stride = layer_params['feat_stride']
        scales = list(map(float, layer_params.get('scales', (8, 16, 32))))
        ratios = list(map(float, layer_params.get('ratios', (0.5, 1.0, 2.0))))
        anchor_base_size = layer_params.get('anchor_base_size', 16)
        min_bbox_size = float(layer_params.get('min_bbox_size', 16.0))
        max_num_proposals = layer_params.get('max_num_proposals', 6000)
        max_num_rois = layer_params.get('max_num_rois', 1)  # Output the top 1 ROI if max_num_rois is not specified
        iou_threshold_nms = float(layer_params.get('iou_threshold_nms', 0.7))

        return op_adapter.ProposalOp(layer.name,
                                     feat_stride=feat_stride,
                                     scales=scales,
                                     ratios=ratios,
                                     anchor_base_size=anchor_base_size,
                                     min_bbox_size=min_bbox_size,
                                     max_num_proposals=max_num_proposals,
                                     max_num_rois=max_num_rois,
                                     iou_threshold_nms=iou_threshold_nms)


CaffeTranslations.register_translation(CaffeProposalTranslation(),
                                       converter_type('python', 'caffe'),
                                       op_adapter.ProposalOp.TRANSLATION_KEY)
