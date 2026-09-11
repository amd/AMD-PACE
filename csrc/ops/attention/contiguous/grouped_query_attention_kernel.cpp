/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <ops/attention/contiguous/grouped_query_attention_kernel.h>
#include <ops/kernels/kernel_primitive_utils.h>
#include <ops/onednn_utils.h>

typedef long int dim_t;

namespace pace {

namespace kernels {

void grouped_query_attention_kernel(
    at::Tensor& output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor> attn_mask) {
  stream engine_stream(cpu_eng());

  memory input_Q_mem = view_tensor_as_memory(query); // b, num_q_heads, s, h
  memory input_K_mem = view_tensor_as_memory(key); // b, num_kv_heads, s, h
  memory input_V_mem = view_tensor_as_memory(value); // b, num_kv_heads, s, h
  memory output_mem = view_tensor_as_memory(output);

  memory attn_mask_mem = memory({}, cpu_eng(), DNNL_MEMORY_NONE);
  if (attn_mask.has_value())
    attn_mask_mem = view_tensor_as_memory(attn_mask.value());

  auto dta = input_Q_mem.get_desc().get_data_type();

  // get memory descriptors
  auto query_md = input_Q_mem.get_desc();
  auto key_md = input_K_mem.get_desc();
  auto value_md = input_V_mem.get_desc();
  auto output_md = output_mem.get_desc();

  const dim_t B = query_md.get_dims()[0];
  const dim_t num_q_heads = query_md.get_dims()[1];
  const dim_t S = query_md.get_dims()[2];
  const dim_t H = query_md.get_dims()[3];

  const dim_t num_kv_heads = key_md.get_dims()[1];
  const dim_t L = key_md.get_dims()[2];

  int N_rep = num_q_heads / num_kv_heads;

  float scale = 1 / sqrt(H);

  int softmax_axis = 4;

  // Create 5D view for Q: [B, num_kv_heads, N_rep, S, H]
  memory::dims query_5d_dims = {B, num_kv_heads, N_rep, S, H};
  memory::desc query_5d_md(query_5d_dims, dta, memory::format_tag::abcde);
  memory query_5d(query_5d_md, cpu_eng(), input_Q_mem.get_data_handle());

  // Create 5D view for K: [B, num_kv_heads, 1, L, H]
  memory::dims key_5d_dims = {B, num_kv_heads, 1, L, H};
  memory::desc key_5d_md(key_5d_dims, dta, memory::format_tag::abcde);
  memory key_5d(key_5d_md, cpu_eng(), input_K_mem.get_data_handle());

  // K matmul: need K^T: [B, num_kv_heads, 1, H, L]
  memory::dims k5d_dims = {B, num_kv_heads, 1, H, L};
  memory::desc k5d_md(k5d_dims, dta, memory::format_tag::abced);
  memory key_trans_5d(k5d_md, cpu_eng(), key_5d.get_data_handle());

  // Output of batched matmul: [B, num_kv_heads, N_rep, S, L]
  memory::dims out5d_dims = {B, num_kv_heads, N_rep, S, L};
  memory::desc out5d_md(out5d_dims, dta, memory::format_tag::abcde);
  memory attn_out5d(out5d_md, cpu_eng());

  matmul_primitive(
      query_5d, query_5d_md, key_trans_5d, k5d_md, scale, attn_out5d, out5d_md);

  // TODO: : evaluate - add mask as a post-op of matmul
  if (attn_mask_mem.get_desc().get_size() &&
      attn_mask_mem.get_data_handle() != nullptr) {
    auto mask_md = attn_mask_mem.get_desc();
    memory::dims mask_5d_dims;
    // When the mask is expanded to per-head (dim 1 == num_q_heads), reinterpret
    // it as [B, num_kv_heads, N_rep, S, L] so each query head gets its own
    // mask values (e.g. per-head attention sinks). Otherwise broadcast as
    // [B, 1, 1, S, L] across all heads.
    if (mask_md.get_dims()[1] == num_q_heads) {
      mask_5d_dims = {B, num_kv_heads, N_rep, S, L};
    } else {
      mask_5d_dims = {B, 1, 1, S, L};
    }
    memory::desc mask_5d_md(mask_5d_dims, dta, memory::format_tag::abcde);
    memory mask_5d(mask_5d_md, cpu_eng(), attn_mask_mem.get_data_handle());

    binary_add_primitive(attn_out5d, out5d_md, mask_5d, mask_5d_md);
  }

  // softmax
  softmax_primitive(attn_out5d, out5d_md, softmax_axis);

  // Create 5D view for V: [B, num_kv_heads, 1, L, H]
  memory value_5d(key_5d_md, cpu_eng(), input_V_mem.get_data_handle());

  memory final_output_5d(query_5d_md, cpu_eng(), output_mem.get_data_handle());

  matmul_primitive(
      attn_out5d,
      out5d_md,
      value_5d,
      key_5d_md,
      1.0f,
      final_output_5d,
      query_5d_md);
}

} // namespace kernels

} // namespace pace
