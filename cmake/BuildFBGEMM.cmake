# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

include(ExternalProject)

set(FBGEMM_PROJECT fbgemm)
set(FBGEMM_VERSION v1.2.0)

set(FBGEMM_PROJ_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/${FBGEMM_PROJECT}/src/${FBGEMM_PROJECT})
set(FBGEMM_PROJ_BUILD_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/${FBGEMM_PROJECT}/src/${FBGEMM_PROJECT}-build)

# Download, build and install FBGEMM library
ExternalProject_Add(
  ${FBGEMM_PROJECT}
  GIT_REPOSITORY https://github.com/pytorch/FBGEMM.git
  GIT_TAG ${FBGEMM_VERSION}
  GIT_SUBMODULES "external/asmjit external/cpuinfo"
  PREFIX ${FBGEMM_PROJECT}
  BINARY_DIR ${FBGEMM_PROJ_DIR}
  CONFIGURE_COMMAND
    cmake -DFBGEMM_LIBRARY_TYPE=static
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_INSTALL_PREFIX=${FBGEMM_PROJ_BUILD_DIR}
    -DFBGEMM_BUILD_TESTS=OFF
    -DFBGEMM_BUILD_BENCHMARKS=OFF
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
  UPDATE_COMMAND "")

set(FBGEMM_INCLUDE_DIR ${FBGEMM_PROJ_BUILD_DIR}/include)
set(FBGEMM_STATIC_LIB ${FBGEMM_PROJ_BUILD_DIR}/lib/libfbgemm.a
                      ${FBGEMM_PROJ_BUILD_DIR}/lib/libasmjit.a)
