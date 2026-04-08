/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef PACE_MACROS_H
#define PACE_MACROS_H

#if USE_ZENDNN
#include <zendnn.hpp>
using namespace zendnn;

#define JIT_ARG_SRC ZENDNN_ARG_SRC
#define JIT_ARG_WEIGHTS ZENDNN_ARG_WEIGHTS
#define JIT_ARG_BIAS ZENDNN_ARG_BIAS
#define JIT_ARG_DST ZENDNN_ARG_DST
#define JIT_ARG_SRC_1 ZENDNN_ARG_SRC_1
#define JIT_ARG_ATTR_MULTIPLE_POST_OP ZENDNN_ARG_ATTR_MULTIPLE_POST_OP

#define JIT_MEMORY_NONE ZENDNN_MEMORY_NONE
#define JIT_MEMORY_ALLOCATE ZENDNN_MEMORY_ALLOCATE

#define GET_DIMS(mem_desc, dim) mem_desc.dims()[dim]
#define GET_DATA_TYPE(memory) memory.get_desc().data_type()

#else
#include "dnnl.hpp"
using namespace dnnl;

#define JIT_ARG_SRC DNNL_ARG_SRC
#define JIT_ARG_WEIGHTS DNNL_ARG_WEIGHTS
#define JIT_ARG_BIAS DNNL_ARG_BIAS
#define JIT_ARG_DST DNNL_ARG_DST
#define JIT_ARG_SRC_1 DNNL_ARG_SRC_1
#define JIT_ARG_ATTR_MULTIPLE_POST_OP DNNL_ARG_ATTR_MULTIPLE_POST_OP

#define JIT_MEMORY_NONE DNNL_MEMORY_NONE
#define JIT_MEMORY_ALLOCATE DNNL_MEMORY_ALLOCATE

#define GET_DIMS(mem_desc, dim) mem_desc.get_dims()[dim]
#define GET_DATA_TYPE(memory) memory.get_desc().get_data_type()

#endif

#endif // PACE_MACROS_H
