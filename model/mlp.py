
import torch


class MLP(torch.nn.Module):

    def __init__(self, 
                 in_dim :     int = 512, 
                 hidden_dim : int = 1024, 
                 out_dim :    int = 1):
        super().__init__()

        self.l1 = torch.nn.Linear(in_dim, hidden_dim)
        self.ac = torch.nn.GELU()
        self.l2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.ac(self.l1(x))
        return self.l2(x)