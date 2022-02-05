import torch
import torch.nn as nn

from torch.nn.functional import threshold
from typing import Callable

from kwinners import Kwinners2d


class Residual(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        """ Wrapper layer that computes fn(x)+x

        Args:
            fn (function): function that is risidualised
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.tensor:
        return self.fn(x) + x


def convmixer(
    dim: int,
    depth: int,
    kernel_size: int = 9,
    patch_size: int = 7,
    n_classes: int = 10,
    sparsity: float = 0.025,
):
    """creates a convmixer model with sparsity as seen on 
    https://github.com/tmp-iclr/convmixer/blob/main/convmixer.py
    
    Args:
        dim (int): number of filters
        depth (int): number of blocks 
        kernel_size (int, optional): kernel size. Defaults to 9.
        patch_size (int, optional): patch size. Defaults to 7.
        n_classes (int, optional): size of output vector. Defaults to 10.
        sparsity (float, optional): percentage of values > 0. Defaults to 0.025.

    Returns:
        nn.Module: the Convmixer model
    """
    k = max(4, int(dim * sparsity))
    return nn.Sequential(
        nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            dim, dim, kernel_size, groups=dim, padding=kernel_size // 2
                        ),
                        nn.GELU(),
                        nn.BatchNorm2d(dim),
                    )
                ),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                Kwinners2d(k),
            )
            for i in range(depth)
        ],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )
