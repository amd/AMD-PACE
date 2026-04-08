# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

# Two types of files needs to be included. One which ends with avx512.cpp and
# other which does not end with avx512.cpp. The ones that end in avx512.cpp
# require special compilation flags. This script will filter out the files.

# Get all the headers for the operators.
file(GLOB OPS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
# OPS_CPP_FILES will contain all the operator files.
file(GLOB OPS_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# KERNELS_CPP_FILES will contain all the files which will not end with avx512.cpp
file(GLOB KERNELS_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)
FILE(GLOB LIBXSMM_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/libxsmm_dependency/*.cpp)
list(FILTER KERNELS_CPP_FILES EXCLUDE REGEX ".*avx512.cpp$")

# KERNELS_AVX512_CPP_FILES will contain all the files which end with avx512.cpp
file(GLOB KERNELS_AVX512_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*avx512.cpp)

# Attention backend kernels (organized under attention/ subdirectories)
file(GLOB_RECURSE ATTN_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/attention/*.cpp)
list(FILTER ATTN_CPP_FILES EXCLUDE REGEX ".*avx512.cpp$")
file(GLOB_RECURSE ATTN_AVX512_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/attention/*avx512.cpp)
list(APPEND KERNELS_CPP_FILES ${ATTN_CPP_FILES})
list(APPEND KERNELS_AVX512_CPP_FILES ${ATTN_AVX512_CPP_FILES})

# Set the variables to the parent scope
set(OPS_HEADERS ${OPS_HEADERS} PARENT_SCOPE)
set(OPS_CPP_FILES ${OPS_CPP_FILES} PARENT_SCOPE)
set(KERNELS_CPP_FILES ${KERNELS_CPP_FILES} PARENT_SCOPE)
set(LIBXSMM_CPP_FILES ${LIBXSMM_CPP_FILES} PARENT_SCOPE)
set(KERNELS_AVX512_CPP_FILES ${KERNELS_AVX512_CPP_FILES} PARENT_SCOPE)
