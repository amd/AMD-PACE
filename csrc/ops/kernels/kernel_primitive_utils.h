/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#ifndef KERNEL_PRIMITIVE_UTILS_H
#define KERNEL_PRIMITIVE_UTILS_H

#include <ops/cpu.h>
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

  // Add scale as eltwise linear post-op if needed
  if (scale != 1.f) {
#ifdef USE_ZENDNN
    matmul_post_ops.append_eltwise(1, algorithm::eltwise_linear, scale, 0);
#else // Use OneDNN
    matmul_post_ops.append_eltwise(algorithm::eltwise_linear, scale, 0);
#endif
  }

  // Track the post-op index for binary add
  int binary_post_op_index = 0;

  // Add binary add post-op if requested
  if (use_binary_add) {
#ifdef USE_ZENDNN
    matmul_post_ops.append_binary(1, algorithm::binary_add, binary_add_md);
#else // Use OneDNN
    matmul_post_ops.append_binary(algorithm::binary_add, binary_add_md);
#endif

    binary_post_op_index = static_cast<int>(matmul_post_ops.len()) - 1;
  }

  matmul_attr.set_post_ops(matmul_post_ops);

#ifdef USE_ZENDNN
  // Create operation descriptor
  auto matmul_d = matmul::desc(a_md, b_md, o_md);
  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, cpu_eng());
#else // Use OneDNN
  // Create Primitive descriptor
  auto matmul_pd =
      matmul::primitive_desc(cpu_eng(), a_md, b_md, o_md, matmul_attr);
#endif

  // Create the primitive.
  auto matmul_prim = matmul(matmul_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({JIT_ARG_SRC, a});
  matmul_args.insert({JIT_ARG_WEIGHTS, b});
  matmul_args.insert({JIT_ARG_DST, o});

  // Add binary post-op argument if requested
  if (use_binary_add) {
    matmul_args.insert(
        {JIT_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_index) | JIT_ARG_SRC_1,
         binary_add_mem});
  }

  // Primitive execution: matrix multiplication with optional scale and binary
  // add.
  matmul_prim.execute(engine_stream, matmul_args);

  // Wait for completion
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
#ifdef USE_ZENDNN
    matmul_post_ops.append_eltwise(1, algorithm::eltwise_linear, scale, 0);
#else // Use OneDNN
    matmul_post_ops.append_eltwise(algorithm::eltwise_linear, scale, 0);
#endif // Use OneDNN
  }
  matmul_attr.set_post_ops(matmul_post_ops);

#ifdef USE_ZENDNN
  // Create operation descriptor
  auto matmul_d = matmul::desc(a_md, b_md, o_md);
  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, cpu_eng());

#else // Use OneDNN
  // Create Primitive descriptor
  auto matmul_pd =
      matmul::primitive_desc(cpu_eng(), a_md, b_md, o_md, matmul_attr);

#endif // Use OneDNN

  // Create the primitive.
  auto matmul_prim = matmul(matmul_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({JIT_ARG_SRC, a});
  matmul_args.insert({JIT_ARG_WEIGHTS, b});
  matmul_args.insert({JIT_ARG_DST, o});

  // Primitive execution: matrix multiplication with ReLU.
  matmul_prim.execute(engine_stream, matmul_args);
}

inline void binary_add_primitive(
    memory& a,
    const memory::desc& a_md,
    const memory& b,
    const memory::desc& b_md) {
  stream engine_stream(cpu_eng());

#ifdef USE_ZENDNN
  auto binary_d = zendnn::binary::desc(algorithm::binary_add, a_md, b_md, a_md);

  auto binary_pd = zendnn::binary::primitive_desc(binary_d, cpu_eng());

#else // Use OneDNN
  auto binary_pd = binary::primitive_desc(
      cpu_eng(), algorithm::binary_add, a_md, b_md, a_md);
#endif // Use OneDNN

  // Create the primitive.
  auto binary_prim = binary(binary_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> binary_add_args;
  binary_add_args.insert({JIT_ARG_SRC, a});
  binary_add_args.insert({JIT_ARG_SRC_1, b});
  binary_add_args.insert({JIT_ARG_DST, a});

  // Primitive execution: binary with ReLU.
  binary_prim.execute(engine_stream, binary_add_args);
}

inline void softmax_primitive(memory& a, const memory::desc& a_md, int axis) {
  stream engine_stream(cpu_eng());

#ifdef USE_ZENDNN
  auto softmax_d =
      softmax_forward::desc(prop_kind::forward_inference, a_md, axis);

  // Create primitive descriptor.
  auto softmax_pd = softmax_forward::primitive_desc(softmax_d, cpu_eng());

#else // Use OneDNN
  auto softmax_pd = softmax_forward::primitive_desc(
      cpu_eng(),
      prop_kind::forward_inference,
      algorithm::softmax_accurate,
      a_md,
      a_md,
      axis);
#endif // Use OneDNN

  // Create the primitive.
  auto softmax_prim = softmax_forward(softmax_pd);

  // Primitive arguments. Set up in-place execution by assigning src as DST.
  std::unordered_map<int, memory> softmax_args;
  softmax_args.insert({JIT_ARG_SRC, a});
  softmax_args.insert({JIT_ARG_DST, a});

  // Primitive execution.
  softmax_prim.execute(engine_stream, softmax_args);
}

} // namespace kernels
} // namespace pace

#endif // KERNEL_PRIMITIVE_UTILS_H
