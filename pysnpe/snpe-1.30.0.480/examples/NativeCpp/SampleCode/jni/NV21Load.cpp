//==============================================================================
//
//  Copyright (c) 2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "SNPE/SNPE.hpp"

#include "SNPE/SNPEFactory.hpp"

#include "DlContainer/IDlContainer.hpp"

#include "SNPE/SNPEBuilder.hpp"

#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"
#include "Util.hpp"

std::unique_ptr<zdl::DlSystem::ITensor> loadNV21Tensor (std::unique_ptr<zdl::SNPE::SNPE> & snpe , const char* inputFileListPath){

    std::unique_ptr<zdl::DlSystem::ITensor> input;
    std::ifstream fileListStream(inputFileListPath);
    std::string fileLine;

    while (std::getline(fileListStream, fileLine))
    {
        if (fileLine.empty()) continue;
        zdl::DlSystem::TensorMap outputTensorMap;
        const auto &strList_opt = snpe->getInputTensorNames();
        const auto &strList = *strList_opt;

        if (strList.size() == 1)
        {
         // If the network has a single input, each line represents the input
         // file to be loaded for that input
            std::string filePath(fileLine);
            std::cout << "Processing DNN Input: " << filePath << "\n";
            std::vector<unsigned char> inputVec = loadByteDataFile(filePath);
            const auto &inputShape_opt = snpe->getInputDimensions(strList.at(0));
            const auto &inputShape = *inputShape_opt;
// NV21 file size = (width*height*3)/2
            size_t NV21InputSize = (inputShape[0]*inputShape[1]*3)/2;
            if (inputVec.size() != NV21InputSize)
            {
                std::cerr << "Size of nv21 input file does not match network.\n";
            }

// Specialized tensor factory for nv21
            input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape,inputVec.data(),NV21InputSize);
         }
    }
    return input;
}
