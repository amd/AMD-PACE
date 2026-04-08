/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef AOCL_DLP_KERNELS_H_
#define AOCL_DLP_KERNELS_H_

#include <cstdlib>
#include "aocl_dlp.h"

// Post-operation types for fused operations
enum class PostOpType { NONE = 0, RELU, GELU, SILU, MUL };

// Configuration structure for matmul operations
struct MatmulConfig {
  float alpha; // Scalar for A*B (default: 1.0f)
  float beta; // Scalar for C (default: 0.0f)
  char order; // Row-major 'R' or Column-major 'C' (default: 'R')
  char transa; // Transpose A: 'N' or 'T' (default: 'N')
  char transb; // Transpose B: 'N' or 'T' (default: 'N')
  char mem_format_a; // Memory format for A: 'N' or 'R' (default: 'N')
  char mem_format_b; // Memory format for B: 'N' or 'R' (default: 'N')

  // Default constructor with standard values
  MatmulConfig()
      : alpha(1.0f),
        beta(0.0f),
        order('R'),
        transa('N'),
        transb('N'),
        mem_format_a('N'),
        mem_format_b('N') {}
};

// Generalized matmul with optional bias and post-ops
void aocl_dlp_matmul_bf16(
    const bfloat16* A, // pointer to M x K BFloat16
    const bfloat16* B, // pointer to K x N BFloat16
    bfloat16* C, // pointer to M x N BFloat16 (output)
    unsigned int M,
    unsigned int K,
    unsigned int N,
    const bfloat16* bias = nullptr, // optional bias vector (size N)
    PostOpType post_op = PostOpType::NONE, // post-operation type
    const bfloat16* mul_input = nullptr, // optional input for MUL post-op
    const MatmulConfig* config = nullptr // optional configuration
);

// Returns the required buffer size in bytes for aocl_dlp_reshape_bf16 output.
// Call this first to allocate memory, then pass the buffer to
// aocl_dlp_reshape_bf16.
size_t aocl_dlp_reshape_bf16_buf_size(
    char order,
    char trans,
    char matrix_type,
    unsigned int M,
    unsigned int N);

// Reshape/reorder with parameterized settings. output must point to a buffer
// of at least aocl_dlp_reshape_bf16_buf_size(order, trans, matrix_type, M, N)
// bytes.
void aocl_dlp_reshape_bf16(
    bfloat16* output,
    bfloat16* input,
    char trans,
    unsigned int M,
    unsigned int N,
    int ld,
    char order = 'R', // Row-major 'R' or Column-major 'C' (default: 'R')
    char matrix_type = 'B' // Matrix type: 'A' or 'B' (default: 'B')
);

#endif
