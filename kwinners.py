import torch
import torch.nn as nn


class Kwinners2d(nn.Module):
    # https://github.com/numenta/htmresearch/blob/master/htmresearch/frameworks/pytorch/functions/k_winners.py
    def __init__(self, k: int):
        """Layer that enforces that only k nerons are active. Currently, this 
        means that only k values are none zero, sparse computation for performance
        is not yet implemented
        Reference: https://github.com/numenta/htmresearch/blob/master/htmresearch/frameworks/pytorch/functions/k_winners.py

        Args:
            k (int): number of active neurons
        """
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """takes the k largest values and sets everything else to 0

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: "sparse" output tensor
        """
        threshold = x.kthvalue(x.shape[1] - self.k + 1, dim=1, keepdim=True)[0]
        mask = x < threshold
        return x.masked_fill(mask, 0)
