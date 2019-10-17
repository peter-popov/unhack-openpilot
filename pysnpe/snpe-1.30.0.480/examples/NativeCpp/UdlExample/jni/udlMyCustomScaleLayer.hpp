//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef UDL_MY_CUSTOM_SCALE_HPP
#define UDL_MY_CUSTOM_SCALE_HPP

#include <vector>

#include "DlSystem/IUDL.hpp"
#include "DlSystem/UDLContext.hpp"

#include "MyUdlLayers.hpp"

namespace myudl {

/**
 * Parameters for MyCustomScale layer
 */
struct MyCustomScaleParams {
    CommonLayerParams common_params;

    bool bias_term;
    std::vector<uint32_t> weights_dim;
    std::vector<float> weights_data;
};

class UdlMyCustomScale final : public zdl::DlSystem::IUDL {
public:
    UdlMyCustomScale(const UdlMyCustomScale&) = delete;
    UdlMyCustomScale& operator=(const UdlMyCustomScale&) = delete;

    /**
     * @brief UDLContext by value but it has move operation
     */
    UdlMyCustomScale(zdl::DlSystem::UDLContext context) :
            m_Context(context) {
    }

    /**
     * @brief Setup User's environment.
     *        This is being called by DnnRunTime framework
     *        to let the user opportunity to setup anything
     *        which is needed for running user defined layers
     * @return true on success, false otherwise
     */
    virtual bool setup(void *cookie, size_t insz, const size_t **indim,
                       const size_t *indimsz, size_t outsz, const size_t **outdim,
                       const size_t *outdimsz) override;

    /**
     * Close the instance. Invoked by DnnRunTime to let
     * the user the opportunity to close handels etc...
     */
    virtual void close(void *cookie) noexcept override;

    /**
     * Execute the user defined layer
     * will contain the return value/output tensor
     */
    virtual bool execute(void *cookie, const float **input, float **output)
            override;
private:

    bool ParseMyCustomLayerParams(const void* buffer, size_t size,
            MyCustomScaleParams& params);

    zdl::DlSystem::UDLContext m_Context;
    size_t m_OutSzDim = 0;
    MyCustomScaleParams m_Params;
};

} // ns myudl

#endif // UDL_MY_CUSTOM_SCALE_HPP
