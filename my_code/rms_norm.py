import torch
from sympy.codegen.fnodes import dimension
from torch import nn
from torch import Tensor
from torch.nn import Parameter

from my_code import utils
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super(RMSNorm, self).__init__()
        self.dd_model = d_model
        self.eps = eps
        # 注意这个weight是一维的
        # RMSNorm的调整也是针对输入的每个token的每个d_model维度的表示做一个处理，降低数据规模
        self.weight = Parameter(torch.randn(d_model)).to(utils.get_device())

    def update_weight(self, weight: Tensor):
        # 这个输入的weight是一个只有1个维度的张量
        assert weight.shape[0] == self.dd_model
        self.weight = weight.to(utils.get_device())

    def forward(self, input_tensor: Tensor) -> Tensor:
        # RMSNorm就是针对输入的一个尺度缩放，保持数据之间的差异，减小数据的规模
        input_tensor = input_tensor.to(utils.get_device())
        d_model = input_tensor.size(-1)
        # [batch, sequence, rms_value]
        rms_matrix = (input_tensor.pow(2).sum(dim=-1, keepdim=True) / d_model + self.eps).sqrt()
        # ([batch, sequence, d_model] / [batch, sequence, 1+rms_value]) * [d_model]
        # 最后实际是一个广播，weight是一个一维的张量，然后每个维度的对应元素都乘以这个张量的对应位置的元素
        return input_tensor / rms_matrix * self.weight