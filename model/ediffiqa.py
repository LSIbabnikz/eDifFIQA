
import torch


EDIFFIQA_CONF = {"ediffiqaL": ("./configs/ediffiqaL_config.yaml", "./weights/ediffiqaL.pth"), 
                 "ediffiqaM": ("./configs/ediffiqaM_config.yaml", "./weights/ediffiqaM.pth"),
                 "ediffiqaS": ("./configs/ediffiqaS_config.yaml", "./weights/ediffiqaS.pth")}

class eDifFIQA(torch.nn.Module):
    """ eDifFIQA model consisting of a pretrained FR backbone (CosFace in the original implementation) and a quality regression MLP head.

    Args:
        base_model (torch.nn.Module): FR backbone used for feature extraction.
        mlp (torch.nn.Module): MLP used as a quality regression head.
        return_feat (bool): Flag for returning features, if set to True the model returns (features, qualities) otherwise only the qualities.
    """

    def __init__(self, 
                 backbone_model : torch.nn.Module,
                 quality_head   : torch.nn.Module,
                 return_feat    : bool = True):
        super().__init__()

        self.base_model = backbone_model
        self.mlp = quality_head
        self.return_feat = return_feat

    def forward(self, x):
        feat = self.base_model(x)
        pred = self.mlp(feat)
        if self.return_feat:
            return (feat, pred)
        return pred