/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ops/attention/contiguous/multi_head_attention_kernel.h>
#include <ops/cpu.h>
#include <ops/kernels/kernel_primitive_utils.h>
#include <pace_tensor/pace_aten_interface.h>

// extension includes
#include <utils/kernel_utils.h>
#include <utils/utils.h>

typedef long int dim_t;

namespace pace {

namespace kernels {

memory pace_transpose(const memory& src_mem, std::vector<int> perm) {
  stream engine_stream(cpu_eng());

  auto src_md = src_mem.get_desc();
  auto dta = GET_DATA_TYPE(src_mem);

#ifdef USE_ZENDNN
  auto data_dims = src_md.dims();
  auto ndata_dims = src_md.dims().size();
#else // Use OneDNN
  auto data_dims = src_md.get_dims();
  auto ndata_dims = src_md.get_dims().size();
#endif // Use OneDNN

  memory::dims transposed_dims(ndata_dims);
  memory::dims strides(ndata_dims);
  memory::dim total_stride = 1;
  for (int i = (int)ndata_dims - 1; i >= 0; i--) {
    transposed_dims[i] = data_dims[perm[i]];
    strides[perm[i]] = total_stride;
    total_stride *= data_dims[perm[i]];
  }

  memory dst_mem({data_dims, dta, strides}, cpu_eng(), JIT_MEMORY_ALLOCATE);

  reorder(src_mem, dst_mem)
      .execute(engine_stream, {{JIT_ARG_SRC, src_mem}, {JIT_ARG_DST, dst_mem}});

  return dst_mem;
}

void mha_kernel_impl(
    memory& output_mem,
    const memory& input_Q_mem,
    const memory& input_K_mem,
    const memory& input_V_mem,
    const memory& input_mask_mem,
    const int& use_KQ) {
  stream engine_stream(cpu_eng());

  auto dta = GET_DATA_TYPE(input_Q_mem);

  // get memory descriptors
  auto query_md = input_Q_mem.get_desc();
  auto key_md = input_K_mem.get_desc();
  auto value_md = input_V_mem.get_desc();
  auto mask_md = input_mask_mem.get_desc();
  auto output_md = output_mem.get_desc();

  const dim_t B = GET_DIMS(query_md, 0);
  const dim_t N = GET_DIMS(query_md, 1);
  const dim_t S = GET_DIMS(query_md, 2);
  const dim_t H = GET_DIMS(query_md, 3);

  // KV cache length L
  const dim_t L = GET_DIMS(key_md, 2);

  float scale = 1 / sqrt(H);

  /* PACE MHA offers 4 different experimental paths
    1)use_KQ = 0, Transpose = 0 -
        softmax (Q . (K mem-desc update)) . V

    2)use_KQ = 0, Transpose = 1 -
        softmax (Q . (K physical transpose)) . V

    3)use_KQ = 1, Transpose = 0 -
        softmax (physical transpose ((K . (Q mem-desc update)))) . V

    4)use_KQ = 1, Transpose = 1 -
        softmax (physical transpose ((K . (Q physical transpose)))) . V
  */

  // Default path is - use_KQ = 0, Transpose = 0

  /* This is set to 0 by default to avoid doing physical transpose
     For experimental purpose, it can be tuned to 1
  */
  bool transpose = 0;

  // Softmax is applied on dim 3 corresponding to L dim (BNSL)
  int softmax_axis = 3;

  // 4 MHA approaches implemented below

  /* Method 1: softmax (Q . (K mem-desc update)) . V
    Most optimal and Default path */
  if ((use_KQ == 0) && (transpose == 0)) {
    memory::dims kt_dims = {B, N, H, L};

    // QK' matmul
    memory::dims qk_output_dims = {B, N, S, L};
    auto qk_output_md =
        memory::desc(qk_output_dims, dta, memory::format_tag::abcd);

    auto qk_output_mem = memory(qk_output_md, cpu_eng());

    auto temp_md = memory::desc(kt_dims, dta, memory::format_tag::abdc);

    // Matmul + Add Mask
    if (input_mask_mem.get_desc().get_size() &&
        input_mask_mem.get_data_handle() != nullptr) {
      matmul_binary_add_primitive(
          input_Q_mem,
          query_md,
          input_K_mem,
          temp_md,
          scale,
          qk_output_mem,
          qk_output_md,
          input_mask_mem,
          mask_md,
          true); // Use binary add

    } else { // Only Matmul
      matmul_primitive(
          input_Q_mem,
          query_md,
          input_K_mem,
          temp_md,
          scale,
          qk_output_mem,
          qk_output_md);
    }

    // softmax
    softmax_primitive(qk_output_mem, qk_output_md, softmax_axis);

    // QKV final matmul
    matmul_primitive(
        qk_output_mem,
        qk_output_md,
        input_V_mem,
        value_md,
        1.0f,
        output_mem,
        output_md);

  }

  /* Method 2: softmax (Q . (K physical transpose)) . V */
  else if ((use_KQ == 0) && (transpose == 1)) {
    std::vector<int> Kperm{0, 1, 3, 2};

    memory::dims k_t_output_dims = {B, N, H, L};
    auto key_transpose_md =
        memory::desc(k_t_output_dims, dta, memory::format_tag::abcd);

    auto key_transpose_mem = pace_transpose(input_K_mem, Kperm);

    // QK' matmul
    memory::dims qk_output_dims = {B, N, S, L};
    auto qk_output_md =
        memory::desc(qk_output_dims, dta, memory::format_tag::abcd);

    auto qk_output_mem = memory(qk_output_md, cpu_eng());

    // Matmul + Add Mask
    if (input_mask_mem.get_desc().get_size() &&
        input_mask_mem.get_data_handle() != nullptr) {
      matmul_binary_add_primitive(
          input_Q_mem,
          query_md,
          key_transpose_mem,
          key_transpose_md,
          scale,
          qk_output_mem,
          qk_output_md,
          input_mask_mem,
          mask_md,
          true); // Use binary add

    } else { // Only Matmul
      matmul_primitive(
          input_Q_mem,
          query_md,
          key_transpose_mem,
          key_transpose_md,
          scale,
          qk_output_mem,
          qk_output_md);
    }

    // softmax
    softmax_primitive(qk_output_mem, qk_output_md, softmax_axis);

    // QKV final matmul
    matmul_primitive(
        qk_output_mem,
        qk_output_md,
        input_V_mem,
        value_md,
        1.0f,
        output_mem,
        output_md);

  }

  /* Method 3: softmax (physical transpose ((K . (Q mem-desc update)))) . V */
  else if ((use_KQ == 1) && (transpose == 0)) {
    memory::dims qt_dims = {B, N, H, S};

    // KQ' matmul
    memory::dims kq_output_dims = {B, N, L, S};
    auto kq_output_md =
        memory::desc(kq_output_dims, dta, memory::format_tag::abcd);

    auto kq_output_mem = memory(kq_output_md, cpu_eng());

    auto temp_md = memory::desc(qt_dims, dta, memory::format_tag::abdc);

    matmul_primitive(
        input_K_mem,
        key_md,
        input_Q_mem,
        temp_md,
        scale,
        kq_output_mem,
        kq_output_md);

    // KQ output transpose
    std::vector<int> kqperm{0, 1, 3, 2};

    memory::dims kq_output_t_dims = {B, N, S, L};
    auto kq_output_transpose_md =
        memory::desc(kq_output_t_dims, dta, memory::format_tag::abcd);

    auto kq_output_transpose_mem = pace_transpose(kq_output_mem, kqperm);

    // Add mask
    if (input_mask_mem.get_desc().get_size() &&
        input_mask_mem.get_data_handle() != nullptr)
      binary_add_primitive(
          kq_output_transpose_mem,
          kq_output_transpose_md,
          input_mask_mem,
          mask_md);

    // softmax
    softmax_primitive(
        kq_output_transpose_mem, kq_output_transpose_md, softmax_axis);

    // QKV Matmul
    matmul_primitive(
        kq_output_transpose_mem,
        kq_output_transpose_md,
        input_V_mem,
        value_md,
        1.0f,
        output_mem,
        output_md);

  }

  /* Method 4: softmax (physical transpose ((K . (Q physical transpose)))) . V
   */
  else if ((use_KQ == 1) && (transpose == 1)) {
    std::vector<int> qperm{0, 1, 3, 2};

    memory::dims qt_dims = {B, N, H, S};
    auto query_transpose_md =
        memory::desc(qt_dims, dta, memory::format_tag::abcd);

    auto query_transpose_mem = pace_transpose(input_Q_mem, qperm);

    // KQ' matmul
    memory::dims kq_output_dims = {B, N, L, S};
    auto kq_output_md =
        memory::desc(kq_output_dims, dta, memory::format_tag::abcd);

    auto kq_output_mem = memory(kq_output_md, cpu_eng());

    matmul_primitive(
        input_K_mem,
        key_md,
        query_transpose_mem,
        query_transpose_md,
        scale,
        kq_output_mem,
        kq_output_md);

    // transpose output: BNLS -> BNSL
    std::vector<int> kq_output_perm{0, 1, 3, 2};

    memory::dims kq_output_t_dims = {B, N, S, L};
    auto kq_output_transpose_md =
        memory::desc(kq_output_t_dims, dta, memory::format_tag::abcd);

    auto kq_output_transpose_mem =
        pace_transpose(kq_output_mem, kq_output_perm);

    // Add mask
    if (input_mask_mem.get_desc().get_size() &&
        input_mask_mem.get_data_handle() != nullptr)
      binary_add_primitive(
          kq_output_transpose_mem,
          kq_output_transpose_md,
          input_mask_mem,
          mask_md);

    // softmax
    softmax_primitive(
        kq_output_transpose_mem, kq_output_transpose_md, softmax_axis);

    matmul_primitive(
        kq_output_transpose_mem,
        kq_output_transpose_md,
        input_V_mem,
        value_md,
        1.0f,
        output_mem,
        output_md);
  }
}

void multi_head_attention_kernel(
    at::Tensor& output,
    const at::Tensor& input_Q,
    const at::Tensor& input_K,
    const at::Tensor& input_V,
    const int& use_KQ,
    const c10::optional<at::Tensor>& input_mask) {
  memory input_Q_mem = view_tensor_as_memory(input_Q);
  memory input_K_mem = view_tensor_as_memory(input_K);
  memory input_V_mem = view_tensor_as_memory(input_V);
  memory output_mem = view_tensor_as_memory(output);

  memory input_mask_mem = memory({}, cpu_eng(), JIT_MEMORY_NONE);

  if (input_mask.has_value()) {
    input_mask_mem = view_tensor_as_memory(input_mask.value());
  }

  mha_kernel_impl(
      output_mem,
      input_Q_mem,
      input_K_mem,
      input_V_mem,
      input_mask_mem,
      use_KQ);
}

} // namespace kernels
} // namespace pace
