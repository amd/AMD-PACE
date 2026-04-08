# Ops supported by AMD PACE

The ops in AMD PACE are implemented as PyTorch C++ extensions with [TORCH_LIBRARY_FRAGMENT](https://pytorch.org/cppdocs/api/define_library_8h_1a2eabc7781e58671237d9d0d282ee1814.html). The ops are defined in `csrc/ops`.

### Using the Ops:
To load/use these ops/kernels, load the `torch` and `pace` libraries as follows:
```
import torch
import pace
```

This will dynamically link the ops registered in the `pace` library to the `torch` library. Ops defined in the AMD PACE are listed below.

1. [Linear Ops](#linear-ops)
2. [Rotary Embedding Ops](#rotary-embedding-ops)
3. [Normalization Ops](#normalization-ops)
4. [Embedding Bag Ops](#embedding-ops)
5. [Binary Ops](#binary-ops)
6. [mlp_mlp_fusion](#mlp_mlp_fusion)

# Linear Ops
These Op implements an inner product of input and weight matrices. The input is a 2D tensor of shape `[batch_size, input_features]` and the weight matrix is a 2D tensor of shape `[output_features, input_features]`. Optionally a bias tensor of shape `[output_features]` can be passed. The output is a 2D matrix of shape `[batch_size, output_features]`. The Op is implemented using ZenDNN/OneDNN primitive: `matmul`.

1. [linear](#linear)
2. [linear_relu](#linear_relu)
3. [qlinear](#qlinear)
4. [qlinear_relu](#qlinear_relu)
5. [qlinear_mul_add](#qlinear_mul_add)
6. [qlinear_sigmoid](#qlinear_sigmoid)


### linear
* Operation: `torch.ops.pace.linear`
* Graph node type: `pace::linear`
* PostOps: None
* Input Types Supported: FP32/BF16
* Weight Types Supported: FP32/BF16
* Bias Types Supported: FP32/BF16
* Output Types Supported: FP32/BF16
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
* File: `csrc/ops/kernels/linear.cpp`
* Correctness Verified: Yes
* Note: All the input, weight and bias tensors must be of the same type.

### linear_relu
* Operation: `torch.ops.pace.linear_relu`
* Graph node type: `pace::linear_relu`
* PostOps: ReLU
* Input Types Supported: FP32/BF16
* Weight Types Supported: FP32/BF16
* Bias Types Supported: FP32/BF16
* Output Types Supported: FP32/BF16
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
* File: `csrc/ops/kernels/linear.cpp`
* Correctness Verified: Yes
* Note: All the input, weight and bias tensors must be of the same type.

### qlinear
* Operation: `torch.ops.pace.qlinear`
* Graph node type: `pace::qlinear`
* PostOps: None
* Input Types Supported: QUINT8/QINT8
* Weight Types Supported: QINT8
* Bias Types Supported: FP32/INT32
* Output Types Supported: FP32/INT8
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
    * `output_scale`: Output scale of dtype double.
    * `output_zero_point`: Output zero point of dtype int.
    * `output_dtype`: Output dtype of the tensor of type torch.dtype. For FP32 output, provide `output_scale` as `1.0` and  `output_zero_point` as `0`.
* File: `csrc/ops/kernels/linear.cpp`

### qlinear_relu
* Operation: `torch.ops.pace.qlinear_relu`
* Graph node type: `pace::qlinear_relu`
* PostOps: ReLU
* Input Types Supported: QUINT8/QINT8
* Weight Types Supported: QINT8
* Bias Types Supported: FP32/INT32
* Output Types Supported: FP32/INT8
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
    * `output_scale`: Output scale of dtype double.
    * `output_zero_point`: Output zero point of dtype int.
    * `output_dtype`: Output dtype of the tensor of type torch.dtype. For FP32 output, provide `output_scale` as `1.0` and  `output_zero_point` as `0`.
* File: `csrc/ops/kernels/linear.cpp`


### qlinear_mul_add
* Operation: `torch.ops.pace.qlinear_mul_add`
* Graph node type: `pace::qlinear_mul_add`
* PostOps: Mul -> Add
* Input Types Supported: QUINT8/QINT8
* Weight Types Supported: QINT8
* Bias Types Supported: FP32/INT32
* Output Types Supported: FP32
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
    * `multiplier`: Multiplier tensor of shape ND `[batch_size, ..., input_features]`.
    * `addend`: Addend tensor of shape ND `[batch_size, ..., input_features]`.
    * `alpha`: Alpha for the addend of type float. Only 1 is supported for now.
* File: `csrc/ops/kernels/linear.cpp`

### qlinear_sigmoid
* Operation: `torch.ops.pace.qlinear_sigmoid`
* Graph node type: `pace::qlinear_sigmoid`
* PostOps: Sigmoid
* Input Types Supported: QUINT8/QINT8
* Weight Types Supported: QINT8
* Bias Types Supported: FP32/INT32
* Output Types Supported: FP32
* Arguments:
    * `input`: Input tensor of shape ND `[batch_size, ..., input_features]`.
    * `weight`: Weight tensor of shape 2D `[output_features, input_features]`.
    * `bias`: Bias tensor of shape 1D `[output_features]`.
* File: `csrc/ops/kernels/linear.cpp`

# Rotary Embedding Ops
These ops apply Rotary Position Embeddings (RoPE) to query and key tensors. The fused kernel applies neox-style RoPE to both Q and K in a single OMP-parallel pass per tensor, avoiding the 6 intermediate tensor allocations of the Python chunk/mul/cat approach.

1. [fused_rope](#fused_rope)

### fused_rope
* Operation: `torch.ops.pace.fused_rope_apply`
* Graph node type: `pace::fused_rope_apply`
* PostOps: None
* Input Types Supported: BF16
* Output Types Supported: BF16
* Arguments:
    * `query`: Query tensor of shape 4D `[BS, num_heads, seq_len, head_dim]` (BNSH) or `[BS, seq_len, num_heads, head_dim]` (BSNH).
    * `key`: Key tensor of same layout as `query`.
    * `cos`: Cosine tensor of shape `[BS, seq_len, head_dim // 2]`.
    * `sin`: Sine tensor of shape `[BS, seq_len, head_dim // 2]`.
    * `unsqueeze_dim`: 1 for BNSH layout, 2 for BSNH layout (int).
* Returns: Tuple of `(query_out, key_out)` with RoPE applied.
* File: `csrc/ops/rope.cpp`, `csrc/ops/kernels/fused_rope.h`
* Correctness Verified: Yes
* Note: `head_dim` must be even. The Python `RotaryEmbedding.apply_rotary_emb` method automatically dispatches to this fused kernel when inputs are contiguous BF16 neox-style.

# Normalization Ops
These ops implement normalization operations using a pure-fused AVX-512 kernel with OMP parallelism. Each row is processed in two vectorized passes: Pass 1 accumulates statistics (and fuses the residual add for the fused variants) entirely in fp32, Pass 2 normalizes and scales the output. A thread-local fp32 scratch buffer avoids bf16 round-trips in the fused path. All four ops share a single templatized kernel (`norm_impl<IsRMSNorm, IsFusedResidual>`) in `csrc/ops/kernels/fused_norm_kernel_avx512.cpp`.

1. [rmsnorm](#rmsnorm)
2. [fused_add_rmsnorm](#fused_add_rmsnorm)
3. [layernorm](#layernorm)
4. [fused_add_layernorm](#fused_add_layernorm)

### rmsnorm
* Operation: `torch.ops.pace.rmsnorm`
* Graph node type: `pace::rmsnorm`
* PostOps: None
* Input Types Supported: BF16
* Weight Types Supported: BF16
* Output Types Supported: BF16
* Arguments:
    * `x`: Input tensor of shape ND `[batch_size, ..., hidden_size]`.
    * `weight`: Scale tensor of shape 1D `[hidden_size]`.
    * `eps`: Epsilon for numerical stability (float).
* File: `csrc/ops/norm.cpp`, `csrc/ops/kernels/fused_norm_kernel.h`
* Correctness Verified: Yes

### fused_add_rmsnorm
* Operation: `torch.ops.pace.fused_add_rmsnorm`
* Graph node type: `pace::fused_add_rmsnorm`
* PostOps: Binary Add (residual)
* Input Types Supported: BF16
* Weight Types Supported: BF16
* Output Types Supported: BF16
* Arguments:
    * `x`: Input tensor of shape ND `[batch_size, ..., hidden_size]`.
    * `residual`: Residual tensor of same shape as `x`.
    * `weight`: Scale tensor of shape 1D `[hidden_size]`.
    * `eps`: Epsilon for numerical stability (float).
* Returns: Tuple of `(normed_output, residual_output)` where `residual_output = x + residual`.
* File: `csrc/ops/norm.cpp`, `csrc/ops/kernels/fused_norm_kernel.h`
* Correctness Verified: Yes

### layernorm
* Operation: `torch.ops.pace.layernorm`
* Graph node type: `pace::layernorm`
* PostOps: None
* Input Types Supported: BF16
* Weight Types Supported: BF16
* Bias Types Supported: BF16
* Output Types Supported: BF16
* Arguments:
    * `x`: Input tensor of shape ND `[batch_size, ..., hidden_size]`.
    * `weight`: Scale tensor of shape 1D `[hidden_size]`.
    * `bias`: Shift tensor of shape 1D `[hidden_size]`.
    * `eps`: Epsilon for numerical stability (float).
* File: `csrc/ops/norm.cpp`, `csrc/ops/kernels/fused_norm_kernel.h`
* Correctness Verified: Yes

### fused_add_layernorm
* Operation: `torch.ops.pace.fused_add_layernorm`
* Graph node type: `pace::fused_add_layernorm`
* PostOps: Binary Add (residual)
* Input Types Supported: BF16
* Weight Types Supported: BF16
* Bias Types Supported: BF16
* Output Types Supported: BF16
* Arguments:
    * `x`: Input tensor of shape ND `[batch_size, ..., hidden_size]`.
    * `residual`: Residual tensor of same shape as `x`.
    * `weight`: Scale tensor of shape 1D `[hidden_size]`.
    * `bias`: Shift tensor of shape 1D `[hidden_size]`.
    * `eps`: Epsilon for numerical stability (float).
* Returns: Tuple of `(normed_output, residual_output)` where `residual_output = x + residual`.
* File: `csrc/ops/norm.cpp`, `csrc/ops/kernels/fused_norm_kernel.h`
* Correctness Verified: Yes

# Embedding Ops
The embedding bag ops

1. [qmerged_embedding_bag_nbit_cat](#qmerged_embedding_bag_nbit_cat)

### qmerged_embedding_bag_nbit_cat
*Note: This method is to be not called directly, this is to be used within AMD PACE if it finds the appropriate pattern.*
* Operation: `torch.ops.pace.qmerged_embedding_bag_nbit_cat`
* Graph node type: `pace::qmerged_embedding_bag_nbit_cat`
* PostOps: Implicit Concat
* Index Type Supported: INT64/INT32
* Offset Type Supported: INT64/INT32
* Weight Types Supported: QINT8/QINT4x2
* Output Types Supported: FP32
* Arguments:
    * `weights`: A vector of size `num_tables` with type `EmbeddingPackedParamsBase`.
    * `indices`: Indices tensor of shape `[batch_size, num_indices]`.
    * `offsets`: Indices tensor of shape `[batch_size + 1, num_indices]`.
    * `dense`: Dense input tensor to be concatenated with the embedding output of shape `[batch_size, embedding_dim]`.
    * `bit_width`: Bit width of weights. Can either be `8` or `4`.
* File: `csrc/ops/kernels/embedding_bag.cpp`
* Extended operator from [PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.2.0/aten/src/ATen/native/quantized/cpu/qembeddingbag.cpp#L396).

# Binary Ops
The binary ops

1. [qmul_add](#qmul_add)

### qmul_add
*Note: This method is to be not called directly, this is to be used within AMD PACE if it finds the appropriate pattern.*

This operator implements fused qmul -> qadd. It can take it combination of inputs as mentioned below. This operator implements a special connection in DLRMv2 model to improve accuracy without compromising on performance.
* Operation: `torch.ops.pace.qmul_add`
* Graph node type: `pace::qmul_add`
* Multiplier Type Supported: FP32
* Multiplicand Type Supported: INT8
* Addend Types Supported: INT8/FP32
* Output Types Supported: INT8
* Arguments:
    * `a`: Tensor of shape `MxN` where 96 is a factor of N.
    * `b`: Tensor of shape `MxN` where 96 is a factor of N.
    * `addend`: Tensor of shape `MxN` where 96 is a factor of N.
    * `o_scale`: Output scale of dtype double.
    * `o_zero_point`: Output zero point of dtype int.
    * `o_dtype`: Output dtype of the tensor of type torch.dtype.
* File: `csrc/ops/kernels/binary.cpp`


### mlp_mlp_fusion
*Note: This method is to be not called directly, this is to be used within AMD PACE if it finds the appropriate pattern.*
This operator implements fused linear + linear for the FFN layer of transformers. It can currently take in the following formats of inputs. This operator implements the IMBPS mlp flow to control the intermediate buffer memory between the two mlp's. Effectively creating the two as a joint operation at cache level.(similar to what postops do at a register level) refer the IMBPS poster for more info.

* Operation it aims to replace LLamaMLP/ OPTDecoderMLP(not a specific function) in HF/vLLM. :  `torch.ops.pace.mlp_mlp_fusion`
* Data Type support configs(src_f32/bf16 weightsMLP1_f32/bf16 inter_activation_f32/bf16 weightsMLP2_f32/bf16 final_activation_f32/bf16)
* Arguments:
    * `src` Src tensor to the first matmul opearation
    * `weight` Array of weight tensors to the first matmul opearation
    * `bias` Array of Bias tensors to be added to first matmul operation
    * `weights2` Array of weights tensors of the second matmul operation
    * `bias2` Bias tensor to the second matmul operation
    * `nlf` Non linearity function
    * `weights_gateProj` Array of weight tensors to the gate projection operation
    * `bias_gateProj` Array of Bias tensors to be added to gate projection operation

* File: `csrc/ops/kernels/mlp_kernel.cpp`