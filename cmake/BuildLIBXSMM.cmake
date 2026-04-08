# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
set(LIBXSMM_PROJECT libxsmm)
set(LIBXSMM_VERSION c14cbc6f8bc7964f8c5190a3a16b8cace03e5889) #TODO: Use the latest version of TPP and add libxsmm_dependency during build
include(ExternalProject)
set(LIBXSMM_PROJ_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/${LIBXSMM_PROJECT}/src/${LIBXSMM_PROJECT})
set(LIBXSMM_PROJ_BUILD_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/${LIBXSMM_PROJECT}/src/${LIBXSMM_PROJECT}-build)
ExternalProject_Add(
  ${LIBXSMM_PROJECT}
  GIT_REPOSITORY https://github.com/libxsmm/libxsmm.git
  GIT_TAG ${LIBXSMM_VERSION}
  PREFIX ${LIBXSMM_PROJECT}
  BINARY_DIR ${LIBXSMM_PROJ_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make -C ${LIBXSMM_PROJ_DIR} NO_BLAS=1 -j
  INSTALL_COMMAND make -C ${LIBXSMM_PROJ_DIR} install PREFIX=${LIBXSMM_PROJ_BUILD_DIR} STATIC=1 NO_BLAS=1
  UPDATE_COMMAND "")
set(LIBXSMM_INCLUDE_DIR ${LIBXSMM_PROJ_BUILD_DIR}/include ${LIBXSMM_PROJ_BUILD_DIR}/include/utils)
set(LIBXSMM_STATIC_LIB
${LIBXSMM_PROJ_BUILD_DIR}/lib/libxsmm.a
)
