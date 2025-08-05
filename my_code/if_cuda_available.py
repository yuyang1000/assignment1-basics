import torch

def test_numpy():
    import torch
    import numpy as np


    np.random.seed(0)

    # 创建一个tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    # 尝试转换为numpy
    x_np = x.detach().cpu().numpy()
    print(x_np)

if __name__ == '__main__':
    # print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    test_numpy()
