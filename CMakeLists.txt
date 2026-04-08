# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(${PACKAGE_NAME} VERSION ${PACKAGE_VERSION})

set(LIBRARY_NAME ${PACKAGE_NAME})
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find required open libraries
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

# Clone and build external dependencies
include(cmake/BuildLIBXSMM.cmake)
include(cmake/BuildJITBackend.cmake)
include(cmake/BuildFBGEMM.cmake)
include(cmake/ClangFormat.cmake)
include(cmake/BuildAOCLDLP.cmake)

set(CMAKE_INSTALL_RPATH $ORIGIN)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/csrc)
