from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
from torch._C import device
from torch.ao.quantization.utils import weight_is_statically_quantized


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    # embedding的每一行都是一个token_id的隐藏张量，从0一直到词表大小
    # 对于token_id=100,那么需要构造一个行向量，其中第99个元素为1，这样与embedding的矩阵相乘就可以得到对应的
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = weights.to(device)
    # 需要构造一个[batch、sequence、vocab]的标识矩阵
    mask = torch.zeros(token_ids.size(0), token_ids.size(1), vocab_size).to(device)
    # [2, 0, 1] 这是一个tokenId的示例（batch, sequence）所以是一个2维的张量
    # 下面就是需要构建的onehot方阵
    # [0,0,1]
    # [1,0,0]
    # [0,1,0]
    for batch_index in range(token_ids.size(0)):
        for token_index in range(token_ids.size(1)):
            # 通过batch_index拿到batch,通过token_index拿到对应的token_id，认为token_id也是从0开始的
            token_id = token_ids[batch_index][token_index]
            # mask矩阵对应的batch的对应tokenIndex选择token_id数值的词
            mask[batch_index][token_index][token_id] = 1
    #  [batch、sequence、vocab][vocab、model] = [batch、sequence、model]
    return mask @ weights


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    from my_code import utils
    w1_weight = w1_weight.to(utils.get_device())
    w2_weight = w2_weight.to(utils.get_device())
    w3_weight = w3_weight.to(utils.get_device())
    in_features = in_features.to(utils.get_device())

    w1_x = in_features @ w1_weight.T # [batch、sequence、d_model] [d_model, d_ff]
    w3_x = in_features @ w3_weight.T # [batch、sequence、d_model] [d_model, d_ff]
    si_lu = w1_x * torch.sigmoid(w1_x) # [batch、sequence、d_ff]
    inner = si_lu * w3_x # [batch、sequence、d_ff]
    return (inner @ w2_weight.T).to(torch.device("cpu"))


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    import math
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Q = Q.to(device) # batch、sequence、d_q
    K = K.to(device) # batch、sequence、d_k
    V = V.to(device) # batch、sequence、d_v
    mask = mask.to(device) # batch、sequence、？

    d_k = Q.size(-1)
    # 矩阵维度是(batch\sequence\sequence)
    after_softmax = torch.softmax(
        (Q @ K.transpose(-1, -2) / math.sqrt(d_k))
        .masked_fill(~mask, float("-inf")), dim=-1)
    # 大多数注意力实现把 mask 定义为 “哪些位置可以被看见 / 保留”，也就是mask里面为False的都是需要被替换的
    # masked_fill是根据布尔张量将为True的位置填充，所以需要取反
    # 这里mask的维度是[4,12,16]为啥不是一个方阵呢？原因是这里的KQV的第2个维度就不一样，正常情况下应该是一样的吧？
    return after_softmax @ V


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    # todo: 首先要熟悉计算流程，熟悉公式，然后是熟悉计算顺序，熟悉torch的一些做法，比如view只可以用在连续的张量上，比如转置.T是默认全都转置
    # todo: view是只可以拆解合并相邻的维度

    # 张量搬到GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    q_proj_weight = q_proj_weight.to(device)
    k_proj_weight = k_proj_weight.to(device)
    v_proj_weight = v_proj_weight.to(device)
    o_proj_weight = o_proj_weight.to(device)
    in_features = in_features.to(device)

    batch_size = in_features.shape[0]
    sequence_length = in_features.shape[1]
    d_k = q_proj_weight.shape[-2]
    per_head_dk = d_k // num_heads
    d_v = v_proj_weight.shape[-2]
    per_head_dv = d_v // num_heads

    # 相乘得到对应的参数 (b\s\d_k)(b\s\d_v)
    # 多头后每个头的输入维度不变，但每个头的d_k或者d_v维度做了拆分（转置前是行，转置后是列）
    Q = ((in_features @ q_proj_weight.T).view(batch_size, sequence_length, num_heads, per_head_dk)
         .transpose(1, 2))
    K = ((in_features @ k_proj_weight.T).view(batch_size, sequence_length, num_heads, per_head_dk)
         .transpose(1, 2))
    V = ((in_features @ v_proj_weight.T).view(batch_size, sequence_length, num_heads, per_head_dv)
         .transpose(1, 2))

    # 开始处理注意力机制部分，也就是个softmax,先计算，然后加上casual_mask，做softmax，
    # 最后after_softmax维度是[batch_size, num_heads, sequence_length, sequence_length]
    import math
    casual_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool().to(device)
    before_softmax = Q @ K.transpose(-1, -2) / math.sqrt(d_k / num_heads)
    before_softmax = before_softmax.masked_fill(casual_mask, float("-inf"))
    after_softmax = torch.softmax(before_softmax, dim=-1)
    # 下面的维度是[batch_size, num_heads, sequence_length, per_head_dv]
    after_v = after_softmax @ V
    # 变换下拼接各个头的输出，这样就变成了[batch_size, sequence_length, dv = num_heads * per_head_dv]
    after_concat = after_v.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
    # 注意最后还是要乘以o的转置才可以，dv要匹配上，最后的输出是[batch_size, sequence_length, d_model]
    return after_concat @ o_proj_weight.T


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def rope2(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    # 向量化实现的 RoPE，支持任意批次维度，性能更佳
    import torch, math

    if d_k % 2 != 0:
        raise ValueError("d_k must be even for RoPE.")

    device, dtype = in_query_or_key.device, in_query_or_key.dtype
    half_dim = d_k // 2

    # 预计算各维度频率 1 / (theta^(2i/d_k))，形状 [half_dim]
    dim_indices = torch.arange(half_dim, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (2 * dim_indices / d_k))

    # token_positions 可能缺少批次维度，这里广播到输入的前置维度
    pos = token_positions.to(device=device, dtype=dtype)
    while pos.ndim < in_query_or_key.ndim - 1:
        pos = pos.unsqueeze(0)

    # 计算旋转角度 (..., seq_len, half_dim)
    angle = pos.unsqueeze(-1) * freqs  # 广播
    cos, sin = torch.cos(angle), torch.sin(angle)

    # 拆分偶数/奇数维度
    even = in_query_or_key[..., 0::2]
    odd = in_query_or_key[..., 1::2]

    # 应用旋转
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos

    # 合并结果
    out = torch.empty_like(in_query_or_key)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out

def rope1(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:

    group_num = d_k // 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    group_tensor = torch.arange(group_num).to(device=device)

    # 这只是当token_index==1的时候的角度
    basic_angles = 1.0 / torch.pow(theta, group_tensor / group_num)  # [d_k//2]
    basic_angles = basic_angles.unsqueeze(0) # [1, 32]
    # token_position的维度是12，最后我们期望的是12 * 32，对于每个位置都有32个角度
    pos = token_positions.to(device=device)

    in_query_or_key = in_query_or_key.to(device)

    while pos.ndim < in_query_or_key.ndim - 1: # 跟输入对齐，输入可以是[batch、head、sequence、last]
        pos = pos.unsqueeze(0) # 先补齐前面的维度
    pos = pos.unsqueeze(-1)    # 然后补齐最后的维度，pos只可以对齐sequence，最后变成[1,  1, 12, 1]，如果是pos是2维的也没问题，很巧妙
    angle = pos * basic_angles # 广播 [1, 12, 1][1, 32] 比较有意思的还是理解广播的概念！！！

    sin, cos = torch.sin(angle), torch.cos(angle) # [1, 12, 32]

    even = in_query_or_key[..., 0::2]  # [..., 12, 32]
    odd = in_query_or_key[..., 1::2]   # [..., 12, 32]

    rotated_even = even * cos - odd * sin # [batch、head、sequence、32]
    rotated_odd = even * sin + odd * cos  # [batch、head、sequence、32]

    out = torch.empty_like(in_query_or_key) # [batch、head、sequence、64]
    out[..., 0::2] = rotated_even # [batch、head、sequence、32]
    out[..., 1::2] = rotated_odd  # [batch、head、sequence、32]
    return out

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.(QK的维度是一样的)
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # return rope2(d_k, theta, max_seq_len, in_query_or_key, token_positions)
    # return rope1(d_k, theta, max_seq_len, in_query_or_key, token_positions)
    from my_code.rope import RoPE

    rope = RoPE(d_k, theta, max_seq_len)
    return rope.forward(in_query_or_key, token_positions)


def apply_rope(tensor, max_seq_len, theta=10000.0):
    """
    Apply Rotary Position Embedding (RoPE) to input tensor.

    Args:
        tensor: Input tensor with shape [batch, num_heads, sequence, d_head]
        max_seq_len: Maximum sequence length
        theta: RoPE base parameter (default: 10000.0)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    import torch
    import math

    batch_size, num_heads, seq_len, d_head = tensor.shape
    device = tensor.device

    # 1. 计算每个维度对的基础角度 θ_i = 1/(theta^(2i/d_head))
    # 只对偶数维度计算，因为RoPE是对维度对(2i, 2i+1)进行旋转
    dim_indices = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)  # [0, 2, 4, ...]
    theta_i = 1.0 / (theta ** (dim_indices / d_head))  # shape: [d_head//2]

    # 2. 计算每个位置的角度 pos * θ_i
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)  # [0, 1, 2, ..., seq_len-1]
    # 外积得到每个位置和每个维度对的角度
    angles = torch.outer(positions, theta_i)  # shape: [seq_len, d_head//2]

    # 3. 计算cos和sin值
    cos_angles = torch.cos(angles)  # [seq_len, d_head//2]
    sin_angles = torch.sin(angles)  # [seq_len, d_head//2]

    # 4. 扩展维度以匹配tensor的shape [batch, num_heads, seq_len, d_head//2]
    cos_angles = cos_angles.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, d_head // 2)
    sin_angles = sin_angles.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, d_head // 2)

    # 5. 将tensor分成偶数和奇数维度
    # tensor[:, :, :, 0::2] 是偶数维度 [batch, num_heads, seq_len, d_head//2]
    # tensor[:, :, :, 1::2] 是奇数维度 [batch, num_heads, seq_len, d_head//2]
    tensor_even = tensor[:, :, :, 0::2]  # x1, x3, x5, ...
    tensor_odd = tensor[:, :, :, 1::2]  # x2, x4, x6, ...

    # 6. 应用RoPE旋转公式
    # 对于每个维度对(x_{2i}, x_{2i+1})，旋转公式为：
    # x'_{2i}   = x_{2i} * cos(θ) - x_{2i+1} * sin(θ)
    # x'_{2i+1} = x_{2i} * sin(θ) + x_{2i+1} * cos(θ)
    rotated_even = tensor_even * cos_angles - tensor_odd * sin_angles
    rotated_odd = tensor_even * sin_angles + tensor_odd * cos_angles

    # 7. 重新组合偶数和奇数维度
    # 创建输出tensor
    output = torch.zeros_like(tensor)
    output[:, :, :, 0::2] = rotated_even  # 偶数位置
    output[:, :, :, 1::2] = rotated_odd  # 奇数位置

    return output


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    import math
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def get_rms_norm(name: str):
        from my_code.rms_norm import RMSNorm
        rms_norm = RMSNorm(d_model=d_model)
        rms_norm.update_weight(weights.get(name))
        return rms_norm

    # ***********************************************************
    # ***********************************************************
    # 1. 先经过rms_norm的处理
    first_rms_norm = get_rms_norm(name="ln1.weight")
    in_features = in_features.to(device=device)
    after_first_norm = first_rms_norm.forward(in_features)
    # ***********************************************************
    # ***********************************************************
    # 3. 经过MHA处理
    mha_input = after_first_norm
    batch_size = in_features.shape[0]
    sequence_length = in_features.shape[1]
    q_weight = weights.get("attn.q_proj.weight").to(device)  # [d_model, d_model]
    k_weight = weights.get("attn.k_proj.weight").to(device)  # [d_model, d_model]
    v_weight = weights.get("attn.v_proj.weight").to(device)  # [d_model, d_model]
    output_proj_weight = weights.get("attn.output_proj.weight").to(device)
    # 下面经过系列操作直接将输入数据维度转成[batch、head_num、sequence、d_model//num_heads]
    Q = (mha_input @ q_weight.T).view(batch_size, sequence_length, num_heads, d_model // num_heads).transpose(1, 2)
    K = (mha_input @ k_weight.T).view(batch_size, sequence_length, num_heads, d_model // num_heads).transpose(1, 2)
    V = (mha_input @ v_weight.T).view(batch_size, sequence_length, num_heads, d_model // num_heads).transpose(1, 2)
    #  [batch、head_num、sequence、d_model//num_heads] [batch、head_num、d_model//num_heads、sequence]
    #  [batch、head_num、sequence、sequence]

    from my_code.rope import RoPE
    rope = RoPE(d_model // num_heads, theta, max_seq_len)
    Q = rope.forward(Q)
    K = rope.forward(K)

    before_softmax = Q @ K.transpose(-1, -2) / math.sqrt(d_model // num_heads)
    mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool().to(device)
    after_mask = before_softmax.masked_fill(mask, float('-inf'))
    after_softmax = torch.softmax(after_mask, dim=-1)
    # [batch、head_num、sequence、d_model//num_heads]
    after_v = after_softmax @ V
    # [batch、sequence、d_model]
    after_concat = after_v.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
    # [batch、sequence、d_model] 维度不一样还好，都是d_model看不出来了
    attention_output = after_concat @ output_proj_weight.T + in_features
    # ***********************************************************
    # ***********************************************************
    # 4. FFN开始之前先做一次Norm
    ffn_input = attention_output
    second_rms_norm = get_rms_norm(name="ln2.weight")
    after_norm2 = second_rms_norm.forward(ffn_input)

    w1 = weights.get("ffn.w1.weight").to(device) # [d_model, d_ff]
    w2 = weights.get("ffn.w2.weight").to(device) # [d_ff, d_model]
    w3 = weights.get("ffn.w3.weight").to(device) # [d_model, d_ff]

    w1_x = after_norm2 @ w1.T # [batch, sequence, d_ff]
    w3_x = after_norm2 @ w3.T
    before_w2 = torch.sigmoid(w1_x) * w1_x * w3_x # [batch, sequence, d_ff]
    ffn_output = before_w2 @ w2.T + attention_output # [batch, sequence, d_model]
    return ffn_output


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    from my_code.rms_norm import RMSNorm
    rms_norm = RMSNorm(d_model=d_model, eps=eps)
    rms_norm.update_weight(weights)
    return rms_norm.forward(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
