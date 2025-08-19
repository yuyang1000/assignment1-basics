from typing import Optional, Callable

import torch
import math

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr'] # get the learning rate
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] # get state associated with p
                t = state.get("t", 0) # get iteration number from the state, or initial value
                grad = p.grad.data # get the gradient of loss with respect to p
                p.data = p.data - lr / math.sqrt(t + 1) * grad # update the tensor in place
                state["t"] = t + 1 # increment iteration number
        return loss


if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=10)

    print(weights)
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"iteration={t}, loss={loss}")
        print(loss.cpu().item())
        loss.backward()
        opt.step()

    print(weights)


