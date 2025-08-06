import torch


class RoPE(torch.nn.Module):

    def __init__(self, d_k, theta, max_seq_len):
        super(RoPE, self).__init__()
        # d_k: XQ的最后维度，其实就是d_k,至于有多少个head,不关心！
        # theta: 角度
        # max_seq_len，sequence的最大长度
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        group_num = d_k // 2

        # 计算下一共有多少组
        group_t = torch.arange(group_num, device=self.device)
        # 如果token_index==1, 那么维度是多少呢？
        self.inv_angles = 1.0 / torch.pow(theta, group_t / group_num) # 维度[group_num,]
        self.inv_angles = self.inv_angles.unsqueeze(0)                # 维度[1, group_num]

    def forward(self, in_query_or_key, token_positions):
        pos = token_positions.to(self.device)  # 最后的维度是[sequence]
        while pos.ndim < in_query_or_key.ndim: # 为了跟输入匹配[1,1,sequence] 前面有几个1并不重要
            pos = pos.unsqueeze(0)

        pos = pos.unsqueeze(-1) # 为了跟输入匹配[1,1,sequence,1] 前面有几个1并不重要
        angles = pos * self.inv_angles.unsqueeze(0) # [1,1,sequence,group_num] ==== [1,1,sequence,1][1,group_num]
        sin, cos = torch.sin(angles).to(self.device), torch.cos(angles).to(self.device)

        even = in_query_or_key[..., 0::2].to(self.device) # [..., sequence, group]
        odd = in_query_or_key[..., 1::2].to(self.device)  # [..., sequence, group]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        out = torch.empty_like(in_query_or_key)
        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd

        return out