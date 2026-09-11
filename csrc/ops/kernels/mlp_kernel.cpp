/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/
// Standard includes
#include <iostream>
#include <unordered_map>
#include <vector>

// extension includes
#include <ops/onednn.h>
#include <ops/onednn_utils.h>
#include <utils/utils.h>

#include <ops/kernels/mlp_kernel.h>
using dt = memory::data_type;
using tag = memory::format_tag;

namespace pace {

namespace kernels {

/* This function implements an optimized approach to improve cache usage
  patterns in multi-layer perceptron (MLP) computations. The optimization is
  inspired by techniques discussed in internal research(Ref. IMBPS Poster). */
void IMBPS_MLP(
    const at::Tensor& src,
    const std::vector<at::Tensor>& weights1,
    const c10::optional<std::vector<at::Tensor>>& bias1,
    const std::vector<at::Tensor>& weights2,
    const c10::optional<at::Tensor>& bias2,
    std::string& nlf,
    memory& dst2_mem,
    const c10::optional<std::vector<at::Tensor>>& weight_gateProj,
    const c10::optional<std::vector<at::Tensor>>& bias_gateProj) {
  stream engine_stream(cpu_eng());
  int iterations = weights1.size();
  std::vector<at::Tensor> weights_gateProj =
      get_optional_list_tensors(weight_gateProj);
  std::vector<at::Tensor> _bias_gateProj =
      get_optional_list_tensors(bias_gateProj);
  std::vector<at::Tensor> bias_upProj = get_optional_list_tensors(bias1);
  at::Tensor bias_downProj = get_bias_from_opt(bias2);

  // Declared here to facilitate execution in Graph-Mode
  memory up_proj_mem_fringe, gateProj_mem_fringe, weights_gateProj_mem_fringe;
  at::Tensor empty_bias2_tensor, empty_bias2_tensor_fringe;

  // Initialize memory objects
  memory src_mem = view_tensor_as_memory(src);
  memory weights_mem = view_tensor_as_memory(weights1[0], transpose_dims(0, 1));
  memory bias_mem;
  if (bias1.has_value()) {
    bias_mem = view_tensor_as_memory(bias_upProj[0]);
  }
  memory weights2_mem =
      view_tensor_as_memory(weights2[0], transpose_dims(0, 1));
  memory bias2_mem;
  if (bias2.has_value()) {
    bias2_mem = view_tensor_as_memory(bias_downProj);
  }
  memory weights_gateProj_mem;
  if (weight_gateProj.has_value())
    weights_gateProj_mem =
        view_tensor_as_memory(weights_gateProj[0], transpose_dims(0, 1));
  memory bias_gateProjmem;
  if (bias_gateProj.has_value()) {
    bias_gateProjmem = view_tensor_as_memory(_bias_gateProj[0]);
  }
  // Get memory descriptors
  memory::desc src_md = src_mem.get_desc();
  memory::desc weights_md = weights_mem.get_desc();
  memory::desc bias_md;
  if (bias1.has_value()) {
    bias_md = bias_mem.get_desc();
  }
  memory::desc weights2_md = weights2_mem.get_desc();
  memory::desc bias2_md;
  if (bias2.has_value()) {
    bias2_md = bias2_mem.get_desc();
  }
  memory::desc weights_gateProj_md;
  if (weight_gateProj.has_value())
    weights_gateProj_md = weights_gateProj_mem.get_desc();
  memory::desc bias_gateProj_md;
  if (bias_gateProj.has_value()) {
    bias_gateProj_md = bias_gateProjmem.get_desc();
  }

  // Get dimensions and Data type
  memory::dims src_dims = {src_md.get_dims()[0], src_md.get_dims()[1]};
  memory::dims weights_dims = {
      weights_md.get_dims()[0], weights_md.get_dims()[1]};
  memory::dims weights_gateProj_dims;
  if (weight_gateProj.has_value())
    weights_gateProj_dims = {
        weights_gateProj_md.get_dims()[0], weights_gateProj_md.get_dims()[1]};
  dt data_type = src_mem.get_desc().get_data_type();

  memory::dim M = src_dims[0], K = src_dims[1], N = weights_dims[1];

  // Destination (dst) tensors dimensions
  memory::dims up_proj_dims = {M, N};
  auto up_proj_md = memory::desc(up_proj_dims, data_type, tag::ab);
  memory::dims down_proj_dims = {M, K};
  auto down_proj_md = memory::desc(down_proj_dims, dt::f32, tag::ab);
  memory up_proj_mem = memory(up_proj_md, cpu_eng(), DNNL_MEMORY_ALLOCATE);
  memory gateProj_mem;
  memory::desc gateProj_md;
  if (weight_gateProj.has_value()) {
    gateProj_md = memory::desc(up_proj_dims, data_type, tag::ab);
    gateProj_mem = memory(gateProj_md, cpu_eng(), DNNL_MEMORY_ALLOCATE);
  }
  // Create empty tensors for bias
  // Added to ensure bias doesn't get added in multiple blocks
  memory bias2_empty_mem;
  empty_bias2_tensor = at::zeros_like(bias_downProj);
  if (bias2.has_value()) {
    bias2_empty_mem = view_tensor_as_memory(empty_bias2_tensor);
  }

  // Prepare primitives and arguments
  std::vector<primitive> prim_vec;
  std::vector<std::unordered_map<int, memory>> prim_args_vec;
  primitive_attr gateProj_attr;
  post_ops gateProj_ops;
  float alpha = 0.f, beta = 0.f;
  if (weight_gateProj.has_value()) {
    alpha = 1.f;
    gateProj_ops.append_eltwise(algorithm::eltwise_swish, alpha, beta);
    gateProj_attr.set_post_ops(gateProj_ops);
  }
  primitive_attr upProj_attr;
  post_ops upProj_ops;
  std::transform(nlf.begin(), nlf.end(), nlf.begin(), ::tolower);
  if (nlf == "gelu") {
    upProj_ops.append_eltwise(algorithm::eltwise_gelu_tanh, alpha, beta);
  } else if (nlf == "relu") {
    upProj_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
  }
  upProj_attr.set_post_ops(upProj_ops);

  primitive_attr downProj_attr;
  post_ops downProj_ops;
  downProj_ops.append_sum(1.f);
  downProj_attr.set_post_ops(downProj_ops);

  matmul::primitive_desc gateProj_pd;
  if (weight_gateProj.has_value())
    gateProj_pd = matmul::primitive_desc(
        cpu_eng(), src_md, weights_gateProj_md, bias_gateProj_md, gateProj_md);
  auto upProj_pd = matmul::primitive_desc(
      cpu_eng(), src_md, weights_md, bias_md, up_proj_md);
  auto downProj_pd = matmul::primitive_desc(
      cpu_eng(),
      up_proj_md,
      weights2_md,
      bias2_md,
      down_proj_md,
      downProj_attr);

  int last_block = iterations - 1;
  bool fringe_block = false;

  if (weights1[0].sizes()[0] != weights1[last_block].sizes()[0]) {
    fringe_block = true;
    iterations--;
  }

  std::vector<memory> result_mems;
  /* TODO: Parallelize this loop(IMBPS_PARALLEL_FLOW)
  at::parallel_for(0, iterations, 1, [&](int64_t begin, int64_t end){ */

  for (int i = 0; i < iterations; ++i) {
    if (weight_gateProj.has_value())
      weights_gateProj_mem =
          view_tensor_as_memory(weights_gateProj[i], transpose_dims(0, 1));
    memory weights_mem =
        view_tensor_as_memory(weights1[i], transpose_dims(0, 1));
    memory bias_mem;
    if (bias1.has_value()) {
      bias_mem = view_tensor_as_memory(bias_upProj[i]);
    }
    memory weights2_mem =
        view_tensor_as_memory(weights2[i], transpose_dims(0, 1));
    memory bias2_mem;
    if (bias2.has_value()) {
      bias2_mem = view_tensor_as_memory(bias_downProj);
    }
    if (bias_gateProj.has_value())
      bias_gateProjmem = view_tensor_as_memory(_bias_gateProj[i]);

    // Primitive arguments for the gateProj linear
    if (weight_gateProj.has_value()) {
      std::unordered_map<int, memory> gateProj_args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_WEIGHTS, weights_gateProj_mem},
          {DNNL_ARG_BIAS, bias_gateProjmem},
          {DNNL_ARG_DST, gateProj_mem}};
      auto gateProj = matmul(gateProj_pd);
      prim_args_vec.push_back(gateProj_args);
      prim_vec.push_back(gateProj);
    }

    // Primitive arguments for the upProj linear
    std::unordered_map<int, memory> upProj_args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, up_proj_mem}};
    auto upProj = matmul(upProj_pd);
    prim_args_vec.push_back(upProj_args);
    prim_vec.push_back(upProj);

    // Primitive arguments for the second linear
    std::unordered_map<int, memory> downProj_args = {
        {DNNL_ARG_SRC, up_proj_mem},
        {DNNL_ARG_WEIGHTS, weights2_mem},
        {DNNL_ARG_BIAS, (i + 1 < iterations) ? bias2_empty_mem : bias2_mem},
        {DNNL_ARG_DST, dst2_mem}};
    auto downProj = matmul(downProj_pd);
    prim_args_vec.push_back(downProj_args);
    prim_vec.push_back(downProj);
  }

  /* Fringe kernel executes the merged last and second to one fringe block,
    When num_blocks does not divide the weights uniformly, we merge the last
    two blocks obtained to have a unequal final block, which facilitates fine
    tuning to fit into L3. */
  if (fringe_block) {
    // Some declarations for GateProjections so they don't go out of scope
    memory::desc weights_gateProj_md;
    memory::dims weights_gateProj_dims;
    memory::desc gateProj_md;

    matmul::primitive_desc gateProj_pd;
    if (weight_gateProj.has_value())
      weights_gateProj_mem_fringe = view_tensor_as_memory(
          weights_gateProj[last_block], transpose_dims(0, 1));
    memory src_mem = view_tensor_as_memory(src);
    memory weights_mem =
        view_tensor_as_memory(weights1[last_block], transpose_dims(0, 1));
    memory bias_mem;
    if (bias1.has_value()) {
      bias_mem = view_tensor_as_memory(bias_upProj[last_block]);
    }
    memory weights2_mem =
        view_tensor_as_memory(weights2[last_block], transpose_dims(0, 1));
    memory bias2_mem;
    if (bias2.has_value()) {
      bias2_mem = view_tensor_as_memory(bias_downProj);
    }
    if (bias_gateProj.has_value())
      bias_gateProjmem = view_tensor_as_memory(_bias_gateProj[last_block]);

    // Get memory descriptors
    if (weight_gateProj.has_value())
      weights_gateProj_md = weights_gateProj_mem_fringe.get_desc();
    memory::desc src_md = src_mem.get_desc();
    memory::desc weights_md = weights_mem.get_desc();
    memory::desc bias_md;
    if (bias1.has_value())
      bias_md = bias_mem.get_desc();
    memory::desc weights2_md = weights2_mem.get_desc();
    memory::desc bias2_md;
    if (bias2.has_value()) {
      bias2_md = bias2_mem.get_desc();
    }
    if (bias_gateProj.has_value())
      bias_gateProj_md = bias_gateProjmem.get_desc();

    // Get dimensions and Data type
    memory::dims src_dims = {src_md.get_dims()[0], src_md.get_dims()[1]};
    memory::dims weights_dims = {
        weights_md.get_dims()[0], weights_md.get_dims()[1]};
    if (weight_gateProj.has_value())
      weights_gateProj_dims = {
          weights_gateProj_md.get_dims()[0], weights_gateProj_md.get_dims()[1]};
    dt data_type = src_mem.get_desc().get_data_type();
    memory::dim M = src_dims[0], K = src_dims[1], N = weights_dims[1];

    // Destination (dst) tensors dimensions
    memory::dims up_proj_dims = {M, N};
    auto up_proj_md = memory::desc(up_proj_dims, data_type, tag::ab);
    memory::dims down_proj_dims = {M, K};
    auto down_proj_md = memory::desc(down_proj_dims, dt::f32, tag::ab);
    up_proj_mem_fringe = memory(up_proj_md, cpu_eng(), DNNL_MEMORY_ALLOCATE);
    if (weight_gateProj.has_value()) {
      gateProj_md = memory::desc(up_proj_dims, data_type, tag::ab);
      gateProj_mem_fringe =
          memory(gateProj_md, cpu_eng(), DNNL_MEMORY_ALLOCATE);
    }

    // Create empty tensors for bias
    // Added to ensure bias doesn't get added in multiple blocks(and still match
    // primitive descriptors)
    empty_bias2_tensor_fringe = at::zeros_like(bias_downProj);
    if (bias2.has_value()) {
      bias2_empty_mem = view_tensor_as_memory(empty_bias2_tensor_fringe);
    }

    primitive_attr gateProj_attr;
    post_ops gateProj_ops;
    float alpha = 0.f, beta = 0.f;
    if (weight_gateProj.has_value()) {
      alpha = 1.f;
      gateProj_ops.append_eltwise(algorithm::eltwise_swish, alpha, beta);
      gateProj_attr.set_post_ops(gateProj_ops);
    }

    primitive_attr upProj_attr;
    post_ops upProj_ops;
    std::transform(nlf.begin(), nlf.end(), nlf.begin(), ::tolower);
    if (nlf == "gelu") {
      upProj_ops.append_eltwise(algorithm::eltwise_gelu_tanh, alpha, beta);
    } else if (nlf == "relu") {
      upProj_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    }
    upProj_attr.set_post_ops(upProj_ops);

    primitive_attr downProj_attr;
    post_ops downProj_ops;
    downProj_ops.append_sum(1.f);
    downProj_attr.set_post_ops(downProj_ops);

    if (weight_gateProj.has_value()) {
      gateProj_pd = matmul::primitive_desc(
          cpu_eng(),
          src_md,
          weights_gateProj_md,
          bias_gateProj_md,
          gateProj_md);
    }
    auto upProj_pd = matmul::primitive_desc(
        cpu_eng(), src_md, weights_md, bias_md, up_proj_md);
    auto downProj_pd = matmul::primitive_desc(
        cpu_eng(),
        up_proj_md,
        weights2_md,
        bias2_md,
        down_proj_md,
        downProj_attr);

    // Primitive arguments for the gateProj linear
    if (weight_gateProj.has_value()) {
      std::unordered_map<int, memory> gateProj_args = {
          {DNNL_ARG_SRC, src_mem},
          {DNNL_ARG_WEIGHTS, weights_gateProj_mem_fringe},
          {DNNL_ARG_BIAS, bias_gateProjmem},
          {DNNL_ARG_DST, gateProj_mem_fringe}};
      auto gateProj = matmul(gateProj_pd);
      prim_args_vec.push_back(gateProj_args);
      prim_vec.push_back(gateProj);
    }

    // Primitive arguments for the upProj linear
    std::unordered_map<int, memory> upProj_args = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, up_proj_mem_fringe}};
    auto upProj = matmul(upProj_pd);
    prim_args_vec.push_back(upProj_args);
    prim_vec.push_back(upProj);

    // Primitive arguments for the second linear
    std::unordered_map<int, memory> downProj_args = {
        {DNNL_ARG_SRC, up_proj_mem_fringe},
        {DNNL_ARG_WEIGHTS, weights2_mem},
        {DNNL_ARG_BIAS, bias2_empty_mem},
        {DNNL_ARG_DST, dst2_mem}};
    auto downProj = matmul(downProj_pd);
    prim_args_vec.push_back(downProj_args);
    prim_vec.push_back(downProj);
  }
  if (!weight_gateProj.has_value()) {
    for (int i = 0; i < prim_vec.size(); i = i + 2) {
      prim_vec.at(i).execute(engine_stream, prim_args_vec.at(i));
      at::Tensor downProj_src_tensor = view_memory_as_tensor(
          prim_args_vec.at(i).at(DNNL_ARG_DST),
          src.scalar_type(),
          transpose_dims(0, 0));
      if (nlf == "gelu")
        downProj_src_tensor = at::gelu(downProj_src_tensor);
      else if (nlf == "relu")
        downProj_src_tensor = at::relu(downProj_src_tensor);
      prim_vec.at(i + 1).execute(
          engine_stream,
          {{DNNL_ARG_SRC, view_tensor_as_memory(downProj_src_tensor)},
           {DNNL_ARG_WEIGHTS, prim_args_vec.at(i + 1).at(DNNL_ARG_WEIGHTS)},
           {DNNL_ARG_BIAS, prim_args_vec.at(i + 1).at(DNNL_ARG_BIAS)},
           {DNNL_ARG_DST, dst2_mem}} /* prim_args_vec.at(i+1)*/);
    }
  } else {
    for (int i = 0; i < prim_vec.size(); i = i + 3) {
      prim_vec.at(i).execute(engine_stream, prim_args_vec.at(i));
      prim_vec.at(i + 1).execute(engine_stream, prim_args_vec.at(i + 1));
      at::Tensor gateProj_tensor = view_memory_as_tensor(
          prim_args_vec.at(i).at(DNNL_ARG_DST),
          src.scalar_type(),
          transpose_dims(0, 0));
      gateProj_tensor = at::silu(gateProj_tensor);
      at::Tensor upProj_tensor = view_memory_as_tensor(
          prim_args_vec.at(i + 1).at(DNNL_ARG_DST),
          src.scalar_type(),
          transpose_dims(0, 0));
      upProj_tensor = upProj_tensor * gateProj_tensor;
      prim_vec.at(i + 2).execute(
          engine_stream,
          {{DNNL_ARG_SRC, view_tensor_as_memory(upProj_tensor)},
           {DNNL_ARG_WEIGHTS, prim_args_vec.at(i + 2).at(DNNL_ARG_WEIGHTS)},
           {DNNL_ARG_BIAS, prim_args_vec.at(i + 2).at(DNNL_ARG_BIAS)},
           {DNNL_ARG_DST, dst2_mem}} /* prim_args_vec.at(i+2)*/);
    }
  }
  // TODO: Parallelize this loop(IMBPS_PARALLEL_FLOW) continuation of line 107
  /* For IMBPS parallel flow */
  // for (int i = 0; i < result_mems.size(); i++) {
  //     auto binary_d = binary::desc(algorithm::binary_add, down_proj_md,
  //     down_proj_md, down_proj_md); auto binary_pd =
  //     binary::primitive_desc(binary_d, cpu_eng()); auto binary_prim =
  //     binary(binary_pd);
  //     // print_data(result_mems[i]);
  //     binary_prim.execute(engine_stream, {
  //         {DNNL_ARG_SRC_0, result_mems[i]},
  //         {DNNL_ARG_SRC_1, dst2_mem},
  //         {DNNL_ARG_DST, dst2_mem}
  //     });

  //     engine_stream.wait();
  // }
}

// TODO: Plan on IMPLEMENTING MLP_MLP_FUSION for int8s.

} // namespace kernels
} // namespace pace
