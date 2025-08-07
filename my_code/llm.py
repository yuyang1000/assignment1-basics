import math

import torch
from torch import nn
from torch import Tensor

from my_code.rms_norm import RMSNorm
from my_code.utils import to_device, get_device


class LLM(nn.Module):
    def __init__(self):
        super(LLM, self).__init__()
        self.embedding: Tensor = None # [vocab_size, d_model]
        self.layers = []
        self.final_norm = None
        self.output_embedding = None
        self.final_softmax = torch.softmax

    def add_layer(self, layer):
        self.layers.append(layer)

    def _get_norm(self, d_model, norm_weight):
        rms_norm = RMSNorm(d_model=d_model)
        rms_norm.update_weight(norm_weight)
        return rms_norm

    def forward(self, input_sequence) -> Tensor:

        # 1. 先做embedding
        after_embedding = self._do_embedding(input_sequence)

        # 2. 然后经过一层又一层的transformer_block
        layer_output = after_embedding
        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        # 3. 最后经过最后的norm
        d_model = after_embedding.size(-1)
        final_norm = self._get_norm(d_model, self.final_norm)
        after_final_norm = final_norm.forward(layer_output)

        # 4. OutputEmbedding [vocab_size, d_model]
        after_final_embedding = after_final_norm @ self.output_embedding.T

        # # 6. SoftMax输出 [batch、sequence、vocab]
        # after_softmax = torch.softmax(after_final_embedding, dim=-1)
        #
        # return after_softmax

        return after_final_embedding

    def _do_embedding(self, input_sequence) -> Tensor:
        assert self.embedding is not None
        # input_sequence的形状[batch、sequence_length]
        # embedding的形状[vocab、d_model]
        # 目标形状：[batch、sequence、d_model]
        # 获取途径 [batch、sequence、vocab][vocab、d_model]
        input_sequence = to_device(input_sequence)
        vocab_size = self.embedding.size(0)
        ones_hot = to_device(torch.zeros(input_sequence.size(0), input_sequence.size(1), vocab_size))
        for batch_idx in range(input_sequence.size(0)):
            for token_idx in range(input_sequence.size(1)): # [100,200,36,45]
                token_value = input_sequence[batch_idx][token_idx]
                ones_hot[batch_idx][token_idx][token_value] = 1
        return ones_hot @ self.embedding


class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.norm1_weight = None
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.o_weight = None
        self.theta = None
        self.num_heads = None

        self.norm2_weight = None
        self.w1_weight = None
        self.w2_weight = None
        self.w3_weight = None

        self.rope = None

    def _get_rope(self, d_k, theta, sequence_length):
        from my_code.rope import RoPE
        return RoPE(d_k, theta, sequence_length)

    def _get_norm(self, d_model, norm_weight):
        rms_norm = RMSNorm(d_model=d_model)
        rms_norm.update_weight(norm_weight)
        return rms_norm

    def forward(self, input_tensor: Tensor) -> Tensor:
        # 输入[batch、sequence、d_model]

        # 1、首先l1_norm
        d_model = input_tensor.size(-1)
        batch_size = input_tensor.size(0)
        sequence_length = input_tensor.size(1)
        d_k = d_model // self.num_heads

        l1_norm = self._get_norm(d_model, self.norm1_weight)
        after_l1_norm = l1_norm.forward(input_tensor)

        # 2、注意力部分 (为啥要转置，因为每个都是d_k * d_model)
        Q = after_l1_norm @ self.q_weight.T
        K = after_l1_norm @ self.k_weight.T
        V = after_l1_norm @ self.v_weight.T # [batch、sequence、d_model]
        # [batch, num_heads, sequence, d_model//num_heads]
        Q = Q.view(batch_size, sequence_length, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, d_model // self.num_heads).transpose(1, 2)

        rope = self._get_rope(d_k, self.theta, sequence_length)
        Q = rope.forward(Q)
        K = rope.forward(K)

        # [batch、num_heads、sequence、sequence]
        before_softmax = Q @ K.transpose(-1, -2) / math.sqrt(d_model // self.num_heads)
        # [sequence, sequence]
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool().to(get_device())
        apply_causal_mask = before_softmax.masked_fill(mask, float('-inf'))
        after_softmax = torch.softmax(apply_causal_mask, dim=-1)
        # [batch、num_heads、sequence、sequence] [batch, num_heads, sequence, d_v]
        # [batch, num_heads, sequence, d_v] -> [batch, sequence, d_model]
        after_concat = (after_softmax @ V).transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        # 这个o默认是d_v * d_model，所以需要转置一下！否则跟上面的d_model对不上
        after_mha = after_concat @ self.o_weight.T + input_tensor

        #*************************************************************************
        # *************************************************************************

        l2_norm = self._get_norm(d_model, self.norm2_weight)
        after_l2_norm = l2_norm.forward(after_mha) # w2: [d_ff, d_model]
        w1_x = after_l2_norm @ self.w1_weight.T      # w1: [d_model, d_ff], [batch, sequence, d_ff]
        w3_x = after_l2_norm @ self.w3_weight.T      # w3: [d_model, d_ff], [batch, sequence, d_ff]

        return (w1_x * torch.sigmoid(w1_x) * w3_x) @ self.w2_weight.T + after_mha