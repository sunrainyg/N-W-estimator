import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        # self.W = nn.Parameter(torch.randn(4,4, dtype = torch.double), requires_grad=True)
        self.W = nn.Parameter(torch.eye(3072, dtype=torch.float32), requires_grad=True).to('cuda')
        # self.W = nn.Parameter(torch.randn(4,4).double(), requires_grad=True) # also works

    def forward(self, x):
        x = torch.matmul(x, self.W.T)
        x = torch.sigmoid(x)
        return x

tnet = TestNet()
print(tnet.W.dtype)
# torch.float64

print(list(tnet.parameters())) 
# [Parameter containing:
# tensor([[-1.9645, -1.5445,  0.2435,  0.4380],
#         [ 1.1403,  0.8836,  0.1811, -0.1212],
#         [ 1.5983, -0.1854, -0.2626,  0.2881],
#         [-1.2364, -0.4802, -0.6038,  0.1164]], requires_grad=True)]
