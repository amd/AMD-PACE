# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
set(AOCLDLP_PROJECT libaocl-dlp)
set(AOCLDLP_VERSION 3256da129b879d09301c966026a603f3f86e1233)
include(ExternalProject)

set(AOCLDLP_PROJ_DIR
	${CMAKE_CURRENT_BINARY_DIR}/${AOCLDLP_PROJECT}/src/${AOCLDLP_PROJECT})
set(AOCLDLP_PROJ_BUILD_DIR
	${CMAKE_CURRENT_BINARY_DIR}/${AOCLDLP_PROJECT}/src/${AOCLDLP_PROJECT}-build)
ExternalProject_Add(
  ${AOCLDLP_PROJECT}
  GIT_REPOSITORY https://github.com/amd/aocl-dlp 
  GIT_TAG ${AOCLDLP_VERSION}
  PREFIX ${AOCLDLP_PROJECT}
  BINARY_DIR ${AOCLDLP_PROJ_DIR}
  CONFIGURE_COMMAND
    cmake -DDLP_THREADING_MODEL=openmp -DCMAKE_INSTALL_PREFIX=${AOCLDLP_PROJ_BUILD_DIR}
  BUILD_COMMAND make -j
  INSTALL_COMMAND make -j install
  UPDATE_COMMAND "")
set(AOCLDLP_INCLUDE_DIR ${AOCLDLP_PROJ_BUILD_DIR}/include)
set(AOCLDLP_STATIC_LIB
${AOCLDLP_PROJ_BUILD_DIR}/lib/libaocl-dlp.a
)
