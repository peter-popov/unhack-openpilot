//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef UDL_MY_UDL_LAYERS_HPP
#define UDL_MY_UDL_LAYERS_HPP

#include <vector>

namespace myudl {

enum LayerType {
    MY_CUSTOM_SCALE_LAYER = 1,
    MY_ANOTHER_LAYER      = 2
};

struct CommonLayerParams {
    LayerType type;
};

/**
 * Parse the common layer parameters from buffer
 * Returns true on success, false on parse error
 */
static bool ParseCommonLayerParams(const void* buffer, size_t size, CommonLayerParams& params) {
    if(!buffer) return false;
    if(size < sizeof(CommonLayerParams)) return false;
    params = *reinterpret_cast<const CommonLayerParams*>(buffer);
    return true;
}

} // ns myudl

#endif // UDL_MY_UDL_LAYERS_HPP
