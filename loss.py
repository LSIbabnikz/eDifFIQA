
import torch

from utils import *

class ExtendedL1(torch.nn.Module):
    """ Implements the extended L1 loss used by the eDifFIQA approach.

    Args:
        epsilon (float): Value of the epsilon parameter of eDifFIQA (default=0.5).
        base_loss (str): Name of the base loss function (default=torch.nn.L1Loss)
    """
    def __init__(self,
                 base_loss: str, 
                 epsilon: float = 0.5):
        super().__init__()

        self.epsilon = epsilon
        self.base_loss = load_module(base_loss)

    def forward(self, input, target, ad_quality=None):
        
        if ad_quality is not None:
            return self.base_loss(input, torch.clamp(target - self.epsilon * ad_quality, 0., 1.))
        else:
            return self.base_loss(input, target)