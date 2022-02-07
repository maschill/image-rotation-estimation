from typing import Tuple
import torch
import torchvision
from torchvision import datasets, transforms
from random import randint


class Cifar10(torch.utils.data.Dataset):
    """Dataloader for MNIST rotation estimation"""

    def __init__(self, train: bool) -> None:
        self.input_channels = 3
        self.mean = 0.1307
        self.std = 0.3081
        self.ds = datasets.CIFAR10(root=".", train=train, download=True)
        self.size = len(self.ds)
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std),]
        )
        self.data = []
        for img, lbl in self.ds:
            if 1:
                self.data.append(img)

    def __len__(self) -> int:
        """returns length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns the randomly rotated image at index idx of the dataset and 
        the (sin(ang), cos(ang)) tuple as a label

        Args:
            idx (int): index of the image

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, label) tuple
        """
        image = self.data[idx]
        ang = torch.rand(1) * 3.1415
        angd = torch.rad2deg(ang)
        image = self.transforms(image)
        image = transforms.functional.rotate(img=image, angle=360.0 - float(angd[0]))
        return image, torch.stack([torch.sin(ang), torch.cos(ang)]).squeeze()

