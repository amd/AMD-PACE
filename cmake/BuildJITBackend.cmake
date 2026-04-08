# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

include(ExternalProject)

set(JIT_BACKEND_PROJECT jit_backend)

if("$ENV{JIT_BACKEND}" STREQUAL "zendnn")

  message(FATAL_ERROR "ZenDNN is not supported with PACE. Please use oneDNN instead.")

else()

  set(JIT_BACKEND_LIB "onednn")
  set(ONEDNN_VERSION v3.11)

  set(ONEDNN_PROJ_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/${JIT_BACKEND_LIB}/src/${JIT_BACKEND_LIB})
  set(ONEDNN_PROJ_BUILD_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/${JIT_BACKEND_LIB}/src/${JIT_BACKEND_LIB}-build)

  # Download, build and install oneDNN library
  ExternalProject_Add(
    ${JIT_BACKEND_PROJECT}
    GIT_REPOSITORY https://github.com/uxlfoundation/oneDNN.git
    GIT_TAG ${ONEDNN_VERSION}
    PREFIX ${JIT_BACKEND_LIB}
    SOURCE_DIR ${ONEDNN_PROJ_DIR}
    BINARY_DIR ${ONEDNN_PROJ_DIR}
    CONFIGURE_COMMAND
    cmake -DONEDNN_LIBRARY_TYPE=STATIC
      -DONEDNN_BUILD_DOC=OFF
      -DONEDNN_BUILD_EXAMPLES=OFF
      -DONEDNN_BUILD_TESTS=OFF
      -DONEDNN_BUILD_GRAPH=OFF
      -DONEDNN_VERBOSE=ON
      -DONEDNN_ENABLE_WORKLOAD=INFERENCE
      -DONEDNN_EXPERIMENTAL_UKERNEL=ON
      -DCMAKE_INSTALL_PREFIX=${ONEDNN_PROJ_BUILD_DIR}
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
    UPDATE_COMMAND "")

  set(JIT_BACKEND_INCLUDE_DIR ${ONEDNN_PROJ_BUILD_DIR}/include/)
  set(JIT_BACKEND_STATIC_LIB ${ONEDNN_PROJ_BUILD_DIR}/lib/libdnnl.a)

endif()
