# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

# Grab all the required files
set(CSRC_ROOT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})
FILE(GLOB CORE_CPP_FILES ${CSRC_ROOT_FOLDER}/core/*.cpp)
FILE(GLOB GRAPH_CPP_FILES ${CSRC_ROOT_FOLDER}/graph/*.cpp)
FILE(GLOB PACETENSOR_CPP_FILES ${CSRC_ROOT_FOLDER}/pace_tensor/*.cpp)
add_subdirectory(${CSRC_ROOT_FOLDER}/ops)

# Merge them into lists
set(ALL_CPP_FILES ${CSRC_ROOT_FOLDER}/torch_extension_bindings.cpp
    ${CORE_CPP_FILES} ${GRAPH_CPP_FILES} ${OPS_CPP_FILES}
    ${KERNELS_CPP_FILES} ${KERNELS_AVX512_CPP_FILES} ${PACETENSOR_CPP_FILES} ${LIBXSMM_CPP_FILES})

# Set the headers to be included
set(PACE_HEADER_FILES ${OPS_HEADERS} ${CORE_HEADERS} ${UTILS_HEADERS})

add_library(${LIBRARY_NAME} SHARED ${ALL_CPP_FILES})
add_dependencies(${LIBRARY_NAME} jit_backend fbgemm libxsmm libaocl-dlp)
target_compile_features(${LIBRARY_NAME} PRIVATE cxx_std_17)

# Automatic clang-format with the flag ENABLE_CLANG_FORMAT
if(ENABLE_CLANG_FORMAT)
  # Collect all source files
  file(GLOB_RECURSE ALL_SOURCE_FILES ${CSRC_ROOT_FOLDER}/*.cpp ${CSRC_ROOT_FOLDER}/*.h)
  # Create a custom target to run clang-format
  set(CLANG_FORMAT_BUILD clang-format-build)
  add_custom_target(
      ${CLANG_FORMAT_BUILD}
      COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${ALL_SOURCE_FILES}
      COMMENT "Running clang-format on the project's sources"
      VERBATIM
  )
add_dependencies(${LIBRARY_NAME} ${CLANG_FORMAT_BUILD})
endif()

# ZenDNN is currently not supported with PACE
# if(JIT_BACKEND_LIB STREQUAL "zendnn")
#   add_definitions(-DUSE_ZENDNN)
# else()
add_definitions(-DUSE_ONEDNN)
# endif()

# AVX512 flags for kernels
set(AVX512_CXX_FLAGS "-mavx2 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16")
set_source_files_properties(${KERNELS_AVX512_CPP_FILES} PROPERTIES COMPILE_FLAGS "${AVX512_CXX_FLAGS}")

# vLLM-derived paged attention files need AVX512 + warning suppression for upstream code
file(GLOB_RECURSE PAGED_ATTN_CPP_FILES ${CSRC_ROOT_FOLDER}/ops/attention/paged/*.cpp)
set(PAGED_ATTN_CXX_FLAGS "${AVX512_CXX_FLAGS} -Wno-unused-variable -Wno-error")
set_source_files_properties(${PAGED_ATTN_CPP_FILES} PROPERTIES COMPILE_FLAGS "${PAGED_ATTN_CXX_FLAGS}")

# Common flags, OMP and TORCH
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -Werror -Wunused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# include dirs
target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${LIBRARY_NAME} PUBLIC ${JIT_BACKEND_INCLUDE_DIR})
target_include_directories(${LIBRARY_NAME} PUBLIC ${FBGEMM_INCLUDE_DIR})
target_include_directories(${LIBRARY_NAME} PUBLIC ${LIBXSMM_INCLUDE_DIR})
target_include_directories(${LIBRARY_NAME} PUBLIC ${AOCLDLP_INCLUDE_DIR})
target_include_directories(${LIBRARY_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(${LIBRARY_NAME} PUBLIC ${PYTHON_INCLUDE_DIR})

# link libraries
target_compile_options(${LIBRARY_NAME} BEFORE PUBLIC ${TORCH_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
target_link_libraries(${LIBRARY_NAME} PRIVATE ${JIT_BACKEND_STATIC_LIB})
target_link_libraries(${LIBRARY_NAME} PRIVATE ${FBGEMM_STATIC_LIB})
target_link_libraries(${LIBRARY_NAME} PRIVATE ${LIBXSMM_STATIC_LIB})
target_link_libraries(${LIBRARY_NAME} PRIVATE ${AOCLDLP_STATIC_LIB})
target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${LIBRARY_NAME} PUBLIC torch_cpu c10)

set_target_properties(${LIBRARY_NAME} PROPERTIES PUBLIC_HEADER "${PACE_HEADER_FILES}")

install(TARGETS ${LIBRARY_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Temporary workaround to install all the header files
# since ATGenie requires them to be in the include folder.
# In the future we will switch to using the PUBLIC_HEADER property above.
set(HEADER_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(PACE_INCLUDE_DIRS ${CSRC_ROOT_FOLDER}/core
                        ${CSRC_ROOT_FOLDER}/graph
                        ${CSRC_ROOT_FOLDER}/ops
                        ${CSRC_ROOT_FOLDER}/pace_tensor
                        ${CSRC_ROOT_FOLDER}/utils)
install(DIRECTORY ${PACE_INCLUDE_DIRS} DESTINATION ${HEADER_INSTALL_DIR} FILES_MATCHING PATTERN "*.h")

get_target_property(TARGET_LINK_LIBRARIES ${LIBRARY_NAME} LINK_LIBRARIES)
message(STATUS "Summary for PACE:")
message(STATUS "Torch version          : ${Torch_VERSION}")
message(STATUS "Torch C++ flags        : ${TORCH_CXX_FLAGS}")
message(STATUS "OpenMP version         : ${OpenMP_CXX_VERSION}")
message(STATUS "OpenMP C++ flags       : ${OpenMP_CXX_FLAGS}")
message(STATUS "CXX flags              : ${CMAKE_CXX_FLAGS}")
message(STATUS "Link libraries         : ${TARGET_LINK_LIBRARIES}")
