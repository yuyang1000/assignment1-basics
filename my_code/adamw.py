import math

import torch


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,):
        # lr: learning rate
        # (β1、β2)：控制动量的估算
        # （λ）weight decay: 权重削减
        defaults = dict(lr=lr)
        self.lr = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=...):
        # 基于梯度，完成参数更新
        # 返回loss
        for param_group in self.param_groups:

            if 't' not in param_group:
                param_group['t'] = 0
            param_group['t'] = param_group['t'] + 1
            params = param_group['params']  # 这是一个tensor的list
            # param_group实际是个dict，针对每个变量，这里尝试在dict里面维护对应的一阶以及二阶动量，先做初始化
            if "first_moment" not in param_group:
                param_group['first_moment'] = [torch.zeros_like(elm) for elm in params]
            if "second_moment" not in param_group:
                param_group['second_moment'] = [torch.zeros_like(elm) for elm in params]
            # 然后就是依次更新每个参数了
            iteration = param_group['t']
            for i in range(len(params)):
                param = params[i]
                if param.grad is None:
                    print(f"skip update parameter for grad is None.")
                    continue
                param_group["first_moment"][i] = self.beta_1 * param_group["first_moment"][i] + ( 1 - self.beta_1) * param.grad
                param_group["second_moment"][i] = self.beta_2 * param_group["second_moment"][i] + (1 - self.beta_2) * (param.grad ** 2)
                lr_t = self.lr * math.sqrt(1 - self.beta_2 ** iteration) / (1 - self.beta_1 ** iteration)
                # Adam更新: θ ← θ - αt * m/√(v+ε)
                adam_update = lr_t * param_group['first_moment'][i] / (self.eps + torch.sqrt(param_group['second_moment'][i]))
                # 权重衰减: θ ← θ - αλθ (应用到原始参数)
                weight_decay_update = self.lr * self.weight_decay * param.data
                # 最终参数更新
                param.data = param.data - adam_update - weight_decay_update
                print(f"after update: {param.data}")

if __name__ == "__main__":
    model = torch.nn.Linear(3, 2, bias=False)
    opt = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    for iteration in range(1000):
        print("*"*30)
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        print(f"iteration:{iteration}, loss: {loss}")
        loss.backward()
        opt.step()
    a = 2
    pass