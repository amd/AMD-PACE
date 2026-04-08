/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include "aocl_dlp_linear_kernel.h"
#include <c10/util/Exception.h>
#include <cstring>

// Helper function to create and configure post-op metadata
static dlp_metadata_t* create_postop_metadata(
    PostOpType post_op,
    unsigned int M,
    unsigned int N,
    const bfloat16* bias,
    const bfloat16* mul_input) {
  if (post_op == PostOpType::NONE && bias == nullptr) {
    return nullptr;
  }

  dlp_metadata_t* metadata = (dlp_metadata_t*)malloc(sizeof(dlp_metadata_t));
  if (metadata == nullptr) {
    return nullptr;
  }

  // Initialize metadata structure
  memset(metadata, 0, sizeof(dlp_metadata_t));

  md_t seq_length = 0;
  DLP_POST_OP_TYPE* seq_vector = (DLP_POST_OP_TYPE*)malloc(
      sizeof(DLP_POST_OP_TYPE) * 2); // Max 2 ops: bias + eltwise

  if (seq_vector == nullptr) {
    free(metadata);
    return nullptr;
  }

  // Add bias if provided
  if (bias != nullptr) {
    dlp_post_op_bias* bias_op =
        (dlp_post_op_bias*)malloc(sizeof(dlp_post_op_bias));
    if (bias_op == nullptr) {
      free(seq_vector);
      free(metadata);
      return nullptr;
    }

    memset(bias_op, 0, sizeof(dlp_post_op_bias));
    bias_op->bias = (void*)bias;
    bias_op->stor_type = DLP_BF16;
    bias_op->sf = nullptr;

    metadata->bias = bias_op;
    seq_vector[seq_length++] = BIAS;
  }

  // Add post-operation based on type
  if (post_op != PostOpType::NONE) {
    dlp_post_op_eltwise* eltwise_op =
        (dlp_post_op_eltwise*)malloc(sizeof(dlp_post_op_eltwise));
    if (eltwise_op == nullptr) {
      if (metadata->bias != nullptr)
        free(metadata->bias);
      free(seq_vector);
      free(metadata);
      return nullptr;
    }

    memset(eltwise_op, 0, sizeof(dlp_post_op_eltwise));
    eltwise_op->sf = nullptr;

    switch (post_op) {
      case PostOpType::RELU:
        eltwise_op->algo.algo_type = RELU;
        eltwise_op->algo.alpha = nullptr;
        eltwise_op->algo.beta = nullptr;
        metadata->eltwise = eltwise_op;
        seq_vector[seq_length++] = ELTWISE;
        metadata->num_eltwise = 1;
        break;

      case PostOpType::GELU:
        eltwise_op->algo.algo_type = GELU_TANH;
        eltwise_op->algo.alpha = nullptr;
        eltwise_op->algo.beta = nullptr;
        metadata->eltwise = eltwise_op;
        seq_vector[seq_length++] = ELTWISE;
        metadata->num_eltwise = 1;
        break;

      case PostOpType::SILU:
        eltwise_op->algo.algo_type = SWISH;
        eltwise_op->algo.alpha = (float*)malloc(sizeof(float));
        eltwise_op->algo.beta = (float*)malloc(sizeof(float));

        // Check for allocation failures
        if (eltwise_op->algo.alpha == nullptr ||
            eltwise_op->algo.beta == nullptr) {
          if (eltwise_op->algo.alpha != nullptr)
            free(eltwise_op->algo.alpha);
          if (eltwise_op->algo.beta != nullptr)
            free(eltwise_op->algo.beta);
          free(eltwise_op);
          if (metadata->bias != nullptr)
            free(metadata->bias);
          free(seq_vector);
          free(metadata);
          return nullptr;
        }

        *((float*)(eltwise_op->algo.alpha)) = (float)1.0;
        *((float*)(eltwise_op->algo.beta)) = (float)1.0;
        eltwise_op->algo.stor_type = DLP_F32;
        metadata->eltwise = eltwise_op;
        seq_vector[seq_length++] = ELTWISE;
        metadata->num_eltwise = 1;
        break;

      case PostOpType::MUL:
        if (mul_input != nullptr) {
          // Matrix multiplication post-op
          dlp_post_op_matrix_mul* mul_op =
              (dlp_post_op_matrix_mul*)malloc(sizeof(dlp_post_op_matrix_mul));
          if (mul_op == nullptr) {
            free(eltwise_op);
            if (metadata->bias != nullptr)
              free(metadata->bias);
            free(seq_vector);
            free(metadata);
            return nullptr;
          }

          memset(mul_op, 0, sizeof(dlp_post_op_matrix_mul));
          mul_op->matrix = (void*)mul_input;
          mul_op->ldm = N; // Leading dimension
          mul_op->stor_type = DLP_BF16;

          // Allocate scale factor structure
          mul_op->sf = (dlp_sf_t*)malloc(sizeof(dlp_sf_t));
          if (mul_op->sf == nullptr) {
            free(mul_op);
            free(eltwise_op);
            if (metadata->bias != nullptr)
              free(metadata->bias);
            free(seq_vector);
            free(metadata);
            return nullptr;
          }

          mul_op->sf->scale_factor = (float*)malloc(sizeof(float));
          if (mul_op->sf->scale_factor == nullptr) {
            free(mul_op->sf);
            free(mul_op);
            free(eltwise_op);
            if (metadata->bias != nullptr)
              free(metadata->bias);
            free(seq_vector);
            free(metadata);
            return nullptr;
          }

          mul_op->sf->scale_factor_len = 1;
          mul_op->sf->scale_factor_type = DLP_F32;
          *((float*)mul_op->sf->scale_factor) = (float)1.0;

          metadata->matrix_mul = mul_op;
          free(eltwise_op); // We don't need eltwise for MUL
          seq_vector[seq_length++] = MATRIX_MUL;
        } else {
          free(eltwise_op);
        }
        break;

      default:
        free(eltwise_op);
        break;
    }
  }

  metadata->seq_length = seq_length;
  metadata->seq_vector = seq_vector;

  return metadata;
}

// Helper function to free post-op metadata
static void free_postop_metadata(dlp_metadata_t* metadata) {
  if (metadata != nullptr) {
    if (metadata->bias != nullptr) {
      free(metadata->bias);
    }
    if (metadata->eltwise != nullptr) {
      // Free nested pointers in eltwise (e.g., alpha and beta for SILU)
      if (metadata->eltwise->algo.alpha != nullptr) {
        free(metadata->eltwise->algo.alpha);
      }
      if (metadata->eltwise->algo.beta != nullptr) {
        free(metadata->eltwise->algo.beta);
      }
      free(metadata->eltwise);
    }
    if (metadata->matrix_mul != nullptr) {
      // Free nested pointers in matrix_mul (e.g., sf and scale_factor)
      if (metadata->matrix_mul->sf != nullptr) {
        if (metadata->matrix_mul->sf->scale_factor != nullptr) {
          free(metadata->matrix_mul->sf->scale_factor);
        }
        free(metadata->matrix_mul->sf);
      }
      free(metadata->matrix_mul);
    }
    if (metadata->seq_vector != nullptr) {
      free(metadata->seq_vector);
    }
    free(metadata);
  }
}

void aocl_dlp_matmul_bf16(
    const bfloat16* A,
    const bfloat16* B,
    bfloat16* C,
    unsigned int M,
    unsigned int K,
    unsigned int N,
    const bfloat16* bias,
    PostOpType post_op,
    const bfloat16* mul_input,
    const MatmulConfig* config) {
  // Use provided config or default values
  MatmulConfig default_config;
  const MatmulConfig& cfg = config ? *config : default_config;

  float alpha = cfg.alpha;
  float beta = cfg.beta;
  char order = cfg.order;
  char transa = cfg.transa;
  char transb = cfg.transb;
  char mem_format_a = cfg.mem_format_a;
  char mem_format_b = cfg.mem_format_b;

  // Calculate leading dimensions based on order and transpose
  md_t lda, ldb, ldc;

  if (order == 'R') { // Row-major
    lda = (transa == 'N') ? K : M;
    ldb = (transb == 'N') ? N : K;
    ldc = N;
  } else { // Column-major
    lda = (transa == 'N') ? M : K;
    ldb = (transb == 'N') ? K : N;
    ldc = M;
  }

  // Create metadata for post-ops and bias
  dlp_metadata_t* postop_metadata =
      create_postop_metadata(post_op, M, N, bias, mul_input);

  // Fail fast if bias/post-ops were requested but metadata allocation failed
  if (bias != nullptr || post_op != PostOpType::NONE) {
    TORCH_CHECK(
        postop_metadata != nullptr,
        "pace::aocl_dlp_matmul_bf16: failed to allocate post-op metadata "
        "(bias or post-op was requested); would produce incorrect output.");
  }

  // Perform the GEMM operation
  aocl_gemm_bf16bf16f32obf16(
      order,
      transa,
      transb,
      M,
      N,
      K,
      alpha,
      A,
      lda,
      mem_format_a,
      B,
      ldb,
      mem_format_b,
      beta,
      C,
      ldc,
      postop_metadata);

  // Cleanup
  free_postop_metadata(postop_metadata);
}

size_t aocl_dlp_reshape_bf16_buf_size(
    char order,
    char trans,
    char matrix_type,
    unsigned int M,
    unsigned int N) {
  msz_t reorder_buffer_size = aocl_get_reorder_buf_size_bf16bf16f32of32(
      order, trans, matrix_type, M, N, NULL);
  return static_cast<size_t>(reorder_buffer_size);
}

void aocl_dlp_reshape_bf16(
    bfloat16* output,
    bfloat16* input,
    char trans,
    unsigned int M,
    unsigned int N,
    int ld,
    char order,
    char matrix_type) {
  // Caller must allocate output with at least aocl_dlp_reshape_bf16_buf_size
  // bytes
  aocl_reorder_bf16bf16f32of32(
      order, trans, matrix_type, input, output, M, N, ld, NULL);
}
