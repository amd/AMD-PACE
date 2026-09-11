# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************
set(AOCLDLP_PROJECT libaocl-dlp)
set(AOCLDLP_VERSION 5.3)
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
      -DCMAKE_INSTALL_LIBDIR=lib
  BUILD_COMMAND make -j
  INSTALL_COMMAND make -j install
  UPDATE_COMMAND "")
set(AOCLDLP_INCLUDE_DIR ${AOCLDLP_PROJ_BUILD_DIR}/include)
set(AOCLDLP_STATIC_LIB
${AOCLDLP_PROJ_BUILD_DIR}/lib/libaocl-dlp.a
)

set(AOCLDLP_LOCALIZED_OBJ ${CMAKE_CURRENT_BINARY_DIR}/aocl_dlp_isolated.o)
set(AOCLDLP_ISOLATE_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/isolate_aocl_xbyak.sh)
find_program(AOCLDLP_LD ld REQUIRED)
find_program(AOCLDLP_OBJCOPY objcopy REQUIRED)
find_program(AOCLDLP_NM nm REQUIRED)
add_custom_command(
  OUTPUT ${AOCLDLP_LOCALIZED_OBJ}
  COMMAND sh ${AOCLDLP_ISOLATE_SCRIPT} ${AOCLDLP_LD} ${AOCLDLP_OBJCOPY}
          ${AOCLDLP_NM} ${AOCLDLP_STATIC_LIB} ${AOCLDLP_LOCALIZED_OBJ}
          ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${AOCLDLP_PROJECT} ${AOCLDLP_ISOLATE_SCRIPT}
  COMMENT "AOCL-DLP: partial-link + privatize bundled Xbyak"
  VERBATIM)
add_custom_target(aocl_dlp_localized DEPENDS ${AOCLDLP_LOCALIZED_OBJ})
