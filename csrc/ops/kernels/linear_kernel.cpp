/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <unordered_map>
#include <vector>

#include <ops/kernels/linear_kernel.h>
#include <ops/onednn.h>
#include <ops/onednn_utils.h>

using tag = memory::format_tag;

namespace pace {

namespace kernels {

void linear_kernel(
    const memory& input,
    memory& weight,
    memory& bias,
    memory& output,
    primitive_attr& attr,
    bool with_reorder) {
  stream s(cpu_eng());

  std::vector<primitive> net_;
  std::vector<std::unordered_map<int, memory>> net_args_;

  tag w_b_format = with_reorder ? tag::any : tag::undef;

  bool with_bias = false;
  if (bias.get_desc().get_size() && bias.get_data_handle() != nullptr) {
    with_bias = true;
  }

  memory::desc weight_desc = desc_from_memory(weight, w_b_format);
  memory::desc bias_desc = memory::desc();
  if (with_bias) {
    bias_desc = desc_from_memory(bias, w_b_format);
  }

  matmul::primitive_desc matmul_prim_desc = matmul::primitive_desc(
      cpu_eng(),
      input.get_desc(),
      weight_desc,
      bias_desc,
      output.get_desc(),
      attr);

  memory weight_reordered = weight;
  if (matmul_prim_desc.weights_desc() != weight.get_desc()) {
    weight_reordered = memory(matmul_prim_desc.weights_desc(), cpu_eng());
    net_.push_back(reorder(weight, weight_reordered));
    net_args_.push_back(
        {{DNNL_ARG_SRC, weight}, {DNNL_ARG_DST, weight_reordered}});
  }

  memory bias_reordered = bias;
  if (with_bias && matmul_prim_desc.bias_desc() != bias.get_desc()) {
    bias_reordered = memory(matmul_prim_desc.bias_desc(), cpu_eng());
    net_.push_back(reorder(bias, bias_reordered));
    net_args_.push_back({{DNNL_ARG_SRC, bias}, {DNNL_ARG_DST, bias_reordered}});
  }

  matmul matmul_prim = matmul(matmul_prim_desc);
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, input});
  matmul_args.insert({DNNL_ARG_WEIGHTS, weight_reordered});
  if (with_bias) {
    matmul_args.insert({DNNL_ARG_BIAS, bias_reordered});
  }
  matmul_args.insert({DNNL_ARG_DST, output});

  net_.push_back(matmul_prim);
  net_args_.push_back(matmul_args);

  for (size_t i = 0; i < net_.size(); ++i) {
    net_.at(i).execute(s, net_args_.at(i));
  }
}

} // namespace kernels

} // namespace pace
