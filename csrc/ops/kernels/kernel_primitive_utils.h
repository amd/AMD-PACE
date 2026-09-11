/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#ifndef KERNEL_PRIMITIVE_UTILS_H
#define KERNEL_PRIMITIVE_UTILS_H

#include <unordered_map>

#include <ops/onednn.h>
#include <utils/utils.h>

namespace pace {

namespace kernels {

inline void matmul_binary_add_primitive(
    const memory& a,
    const memory::desc& a_md,
    const memory& b,
    const memory::desc& b_md,
    const float scale,
    memory& o,
    const memory::desc& o_md,
    const memory& binary_add_mem,
    const memory::desc& binary_add_md,
    const bool use_binary_add = false) {
  stream engine_stream(cpu_eng());

  primitive_attr matmul_attr;
  post_ops matmul_post_ops;

  if (scale != 1.f) {
    matmul_post_ops.append_eltwise(algorithm::eltwise_linear, scale, 0);
  }

  int binary_post_op_index = 0;

  if (use_binary_add) {
    matmul_post_ops.append_binary(algorithm::binary_add, binary_add_md);
    binary_post_op_index = static_cast<int>(matmul_post_ops.len()) - 1;
  }

  matmul_attr.set_post_ops(matmul_post_ops);

  auto matmul_pd =
      matmul::primitive_desc(cpu_eng(), a_md, b_md, o_md, matmul_attr);

  auto matmul_prim = matmul(matmul_pd);

  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, a});
  matmul_args.insert({DNNL_ARG_WEIGHTS, b});
  matmul_args.insert({DNNL_ARG_DST, o});

  if (use_binary_add) {
    matmul_args.insert(
        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_index) | DNNL_ARG_SRC_1,
         binary_add_mem});
  }

  matmul_prim.execute(engine_stream, matmul_args);
  engine_stream.wait();
}

inline void matmul_primitive(
    const memory& a,
    const memory::desc& a_md,
    const memory& b,
    const memory::desc& b_md,
    const float scale,
    memory& o,
    const memory::desc& o_md) {
  stream engine_stream(cpu_eng());

  primitive_attr matmul_attr;
  post_ops matmul_post_ops;

  if (scale != 1.f) {
    matmul_post_ops.append_eltwise(algorithm::eltwise_linear, scale, 0);
  }
  matmul_attr.set_post_ops(matmul_post_ops);

  auto matmul_pd =
      matmul::primitive_desc(cpu_eng(), a_md, b_md, o_md, matmul_attr);

  auto matmul_prim = matmul(matmul_pd);

  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, a});
  matmul_args.insert({DNNL_ARG_WEIGHTS, b});
  matmul_args.insert({DNNL_ARG_DST, o});

  matmul_prim.execute(engine_stream, matmul_args);
}

inline void binary_add_primitive(
    memory& a,
    const memory::desc& a_md,
    const memory& b,
    const memory::desc& b_md) {
  stream engine_stream(cpu_eng());

  auto binary_pd = binary::primitive_desc(
      cpu_eng(), algorithm::binary_add, a_md, b_md, a_md);

  auto binary_prim = binary(binary_pd);

  std::unordered_map<int, memory> binary_add_args;
  binary_add_args.insert({DNNL_ARG_SRC, a});
  binary_add_args.insert({DNNL_ARG_SRC_1, b});
  binary_add_args.insert({DNNL_ARG_DST, a});

  binary_prim.execute(engine_stream, binary_add_args);
}

inline void softmax_primitive(memory& a, const memory::desc& a_md, int axis) {
  stream engine_stream(cpu_eng());

  auto softmax_pd = softmax_forward::primitive_desc(
      cpu_eng(),
      prop_kind::forward_inference,
      algorithm::softmax_accurate,
      a_md,
      a_md,
      axis);

  auto softmax_prim = softmax_forward(softmax_pd);

  std::unordered_map<int, memory> softmax_args;
  softmax_args.insert({DNNL_ARG_SRC, a});
  softmax_args.insert({DNNL_ARG_DST, a});

  softmax_prim.execute(engine_stream, softmax_args);
}

} // namespace kernels
} // namespace pace

#endif // KERNEL_PRIMITIVE_UTILS_H
