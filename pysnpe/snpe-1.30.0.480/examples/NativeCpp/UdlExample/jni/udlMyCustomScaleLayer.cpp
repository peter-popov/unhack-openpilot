//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <iostream>
#include <cstring>
#include <iterator>
#include <numeric>

#include "udlMyCustomScaleLayer.hpp"

namespace {

std::size_t getSizeByDim(const std::vector<size_t>& dim) {
    return std::accumulate(std::begin(dim), std::end(dim), 1,
            std::multiplies<size_t>());
}

} // ns

namespace myudl {

bool UdlMyCustomScale::setup(void *cookie,
                             size_t insz, const size_t **indim, const size_t *indimsz,
                             size_t outsz, const size_t **outdim, const size_t *outdimsz) {

    std::cout << "UdlMyCustomScale::setup() of name " << m_Context.getName()
            << std::endl;

    if (cookie != (void*) 0xdeadbeaf) {
        std::cerr << "UdlMyCustomScale::setup() cookie should be 0xdeadbeaf"
                << std::endl;
        return false;
    }
    if (insz != 1 or outsz != 1) {
        std::cerr << "UdlMyCustomScale::setup() insz=" << insz << " outsz="
                << outsz << std::endl;
        std::cerr
                << "UdlMyCustomScale::setup() multi-input or multi-output not supported"
                << std::endl;
        return false;
    }
    if (indimsz[0] != outdimsz[0]) {
        std::cerr << "UdlMyCustomScale::setup() not the same number of dim, in:"
                << indimsz[0] << " != : " << outdimsz[0] << std::endl;
        return false;
    }
    // compute dims and compare. keep the output dim
    size_t inszdim = getSizeByDim(
            std::vector<size_t>(indim[0], indim[0] + indimsz[0]));
    m_OutSzDim = getSizeByDim(
            std::vector<size_t>(outdim[0], outdim[0] + outdimsz[0]));
    std::cout << "UdlMyCustomScale::setup() input size dim: " << inszdim
            << ", output: " << m_OutSzDim << std::endl;
    if (inszdim != m_OutSzDim) {
        std::cerr << "UdlMyCustomScale::setup() not the same overall dim, in:"
                << inszdim << " != out: " << m_OutSzDim << std::endl;
        return false;
    }
    // parse the params
    const void* blob = m_Context.getBlob();
    std::cout << "UdlMyCustomScale::setup() got blob size "
            << m_Context.getSize() << std::endl;
    if (!blob) {
        std::cerr << "UdlMyCustomScale::setup() got null blob " << std::endl;
        return false;
    }

    if (!ParseMyCustomLayerParams(blob, m_Context.getSize(), m_Params)) {
        std::cerr << "UdlMyCustomScale::setup() failed to parse layer params "
                << std::endl;
        return false;
    }

    // Check the params
    if (m_Params.bias_term) {
        std::cerr << "UdlMyCustomScale::setup() bias term not supported! "
                << std::endl;
        return false;
    }
    if (m_Params.weights_dim.size() != 1
            or m_Params.weights_dim[0] != inszdim) {
        std::cerr << "UdlMyCustomScale::setup() invalid weights" << std::endl;
        return false;
    }

    std::cout << "UdlMyCustomScale::setup() bias_term=" << m_Params.bias_term << std::endl;
    std::cout << "UdlMyCustomScale::setup() weight dimensions: (";
    for(size_t i=0; i<m_Params.weights_dim.size(); i++) {
        std::cout << m_Params.weights_dim[i] << ",";
    }
    std::cout << ")" << std::endl;
    std::cout << "UdlMyCustomScale::setup() # weights=" << m_Params.weights_data.size() << std::endl;

    return true;
}

void UdlMyCustomScale::close(void *cookie) noexcept {
    if (cookie != (void*) 0xdeadbeaf) {
        std::cerr << "UdlMyCustomScale::close() cookie should be 0xdeadbeaf"
                << std::endl;
    }
    std::cout << "UdlMyCustomScale::close()" << std::endl;
    delete this;
}

bool UdlMyCustomScale::execute(void *cookie, const float **input,
                               float **output) {
    if (cookie != (void*) 0xdeadbeaf) {
        std::cerr << "UdlMyCustomScale::execute() cookie should be 0xdeadbeaf"
                << std::endl;
        return false;
    }
    std::cout << "UdlMyCustomScale::execute()" << std::endl;
    for (size_t i = 0; i < m_OutSzDim; i++) {
        output[0][i] = input[0][i] * m_Params.weights_data[i];
    }
    return true;
}

bool UdlMyCustomScale::ParseMyCustomLayerParams(const void* buffer, size_t size,
            MyCustomScaleParams& params) {
    if(!ParseCommonLayerParams(buffer, size, m_Params.common_params)) return false;

    size_t r_size = size - sizeof(CommonLayerParams);
    uint8_t* r_buffer = (uint8_t*) buffer;
    r_buffer += sizeof(CommonLayerParams);

    // bias_term
    if(r_size < sizeof(bool)) return false;
    params.bias_term = *reinterpret_cast<bool*>(r_buffer);
    r_size -= sizeof(bool);
    r_buffer += sizeof(bool);

    // weights_dim
    // packing order:
    //   uint32_t containing # elements
    //   uint32_t[] containing values
    if(r_size < sizeof(uint32_t)) return false;
    uint32_t num_dims = *reinterpret_cast<uint32_t*>(r_buffer);
    r_size -= sizeof(uint32_t);
    r_buffer += sizeof(uint32_t);

    if(r_size < num_dims*sizeof(uint32_t)) return false;
    uint32_t* dims = reinterpret_cast<uint32_t*>(r_buffer);
    params.weights_dim = std::vector<uint32_t>(dims, dims+num_dims);
    r_size -= num_dims*sizeof(uint32_t);
    r_buffer += num_dims*sizeof(uint32_t);

    // weights_data
    // packing order:
    //   uint32_t containing # elements
    //   float[] containins values
    if(r_size < sizeof(uint32_t)) return false;
    uint32_t num_weights = *reinterpret_cast<uint32_t*>(r_buffer);
    r_size -= sizeof(uint32_t);
    r_buffer += sizeof(uint32_t);

    if(r_size < num_weights*sizeof(float)) return false;
    float* weights = reinterpret_cast<float*>(r_buffer);
    params.weights_data = std::vector<float>(weights, weights+num_weights);
    r_size -= num_weights*sizeof(float);
    r_buffer += num_weights*sizeof(float);

    return r_size == 0;
}

} // ns batchrun
