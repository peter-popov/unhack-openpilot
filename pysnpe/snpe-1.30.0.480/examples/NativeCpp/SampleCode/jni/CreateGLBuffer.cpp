//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifdef ANDROID

#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <vector>
#include "CreateGLBuffer.hpp"

void CreateGLContext();

CreateGLBuffer::CreateGLBuffer() {
    this->createGLContext();
}

CreateGLBuffer::~CreateGLBuffer() {
}

void CreateGLBuffer::createGLContext() {
    CreateGLContext();
}

GLuint CreateGLBuffer::convertImage2GLBuffer(const std::vector<std::string>& fileLines, const size_t bufSize)
{
   std::cout << "Processing DNN Input: " << std::endl;
   std::vector<uint8_t> inputVec;
   for(size_t i = 0; i < fileLines.size(); ++i) {
      std::string fileLine(fileLines[i]);
      // treat each line as a space-separated list of input files
      std::vector<std::string> filePaths;
      split(filePaths, fileLine, ' ');
      std::string filePath(filePaths[0]);
      std::cout << "\t" << i + 1 << ") " << filePath << std::endl;
      loadByteDataFileBatched(filePath, inputVec, i);
   }
   GLuint userBuffers;
   glGenBuffers(1, &userBuffers);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, userBuffers);
   glBufferData(GL_SHADER_STORAGE_BUFFER, bufSize, inputVec.data(), GL_STREAM_DRAW);

   return userBuffers;
}

void CreateGLBuffer::setGPUPlatformConfig(zdl::DlSystem::PlatformConfig& platformConfig)
{
    void* glcontext = eglGetCurrentContext();
    void* gldisplay = eglGetCurrentDisplay();
    zdl::DlSystem::UserGLConfig userGLConfig;
    userGLConfig.userGLContext = glcontext;
    userGLConfig.userGLDisplay = gldisplay;
    zdl::DlSystem::UserGpuConfig userGpuConfig;
    userGpuConfig.userGLConfig = userGLConfig;
    platformConfig.setUserGpuConfig(userGpuConfig);
}

#endif
