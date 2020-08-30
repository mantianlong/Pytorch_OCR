import torch
import torch.nn as nn
x = torch.Tensor(100, 16, 5, 5)
x = x.view(x.shape[0], -1)
F = nn.Linear(16 * 5 * 5, 120)
x = F(x)
print(x.size())