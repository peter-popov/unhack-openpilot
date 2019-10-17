//==============================================================================
//
//  Copyright (c) 2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifdef ANDROID

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "Util.hpp"
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include "DlSystem/PlatformConfig.hpp"

#define GL_SHADER_STORAGE_BUFFER          0x90D2

class CreateGLBuffer {

public:
    CreateGLBuffer();
    ~CreateGLBuffer();
    GLuint convertImage2GLBuffer(const std::vector<std::string>& fileLines, const size_t bufSize);
    void setGPUPlatformConfig(zdl::DlSystem::PlatformConfig& platformConfig);

private:
    void createGLContext();

};

#endif
