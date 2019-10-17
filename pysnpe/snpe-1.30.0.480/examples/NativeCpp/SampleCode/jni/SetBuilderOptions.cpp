//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessorOrder(runtimeList)
       .setUdlBundle(udlBundle)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .build();
    return snpe;
}
