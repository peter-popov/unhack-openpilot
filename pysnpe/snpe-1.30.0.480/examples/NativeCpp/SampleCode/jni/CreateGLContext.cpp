//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifdef ANDROID

#include <iostream>
#include <cstdlib>
#include <string>
#include <EGL/egl.h>
#include <GLES2/gl2.h>

#define EGL_RESULT_CHECK(X) do { \
                                   EGLint error = eglGetError(); \
                                   if (!(X) || error != EGL_SUCCESS) { \
                                       std::cerr << \
                                          "EGL error " << error << " at " << __FILE__ << ":" << __LINE__ <<std::endl;\
                                       std::exit(1); \
                                    } \
                            } while (0)


void CreateGLContext() {
    const EGLint attribListWindow[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE
    };
    const EGLint srfPbufferAttr[] = {
        EGL_WIDTH, 512,
        EGL_HEIGHT, 512,
        EGL_LARGEST_PBUFFER, EGL_TRUE,
        EGL_NONE
    };
    static const EGLint gl_context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };

    EGLDisplay eglDisplay = 0;
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGL_RESULT_CHECK(eglDisplay != EGL_NO_DISPLAY);

    EGLint iMajorVersion, iMinorVersion;
    EGL_RESULT_CHECK(eglInitialize(eglDisplay, &iMajorVersion, &iMinorVersion));

    EGLConfig eglConfigWindow = 0;
    int iConfigs = 0;
    EGL_RESULT_CHECK(eglChooseConfig(eglDisplay, attribListWindow, &eglConfigWindow, 1, &iConfigs));

    EGLSurface eglSurfacePbuffer = 0;
    eglSurfacePbuffer = eglCreatePbufferSurface(eglDisplay, eglConfigWindow,srfPbufferAttr);
    EGL_RESULT_CHECK(eglSurfacePbuffer != EGL_NO_SURFACE);

    EGLContext eglContext = 0;
    eglContext = eglCreateContext(eglDisplay, eglConfigWindow, EGL_NO_CONTEXT, gl_context_attribs);
    EGL_RESULT_CHECK(eglContext != EGL_NO_CONTEXT);

    EGL_RESULT_CHECK(eglMakeCurrent(eglDisplay, eglSurfacePbuffer, eglSurfacePbuffer, eglContext));
}

#endif
