"""

Last Update:
    Owner: Kartik M. Jalal
    Date: 21/09/2025
"""

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinUNETR

class SwimUNETR_Heatmap_offsets(nn.Module):
    """
        Wrap SwimUNETR to produce:
            - heatmap logits: (B, 1, Z, Y, X)
            - offsets: (B, 3, Z, Y, X) (Δz,Δy,Δx)
    """

    def __init__(
        self,
        in_channels=1,
        feature_size=48,
        use_checkpoint=True,
        use_v2=True
    ):
        super().__init__()
        self.net = SwinUNETR(
            in_channels=in_channels,
            out_channels=4,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=3, 
            use_v2=use_v2
        )

    def forward(
        self,
        img # (B, 1, Z, Y, X)
    ):
        output_channels = self.net(img) # (B, 4, Z, Y, X)
        heatmap_logits = output_channels[:, :1] # (B, 1, Z, Y, X)
        offsets = output_channels[:, 1:4] # (B, 3, Z, Y, X)

        return heatmap_logits, offsets