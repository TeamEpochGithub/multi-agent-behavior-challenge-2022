import numpy as np
import torch
import torchvision
from simclr import SimCLR
import torchvision.transforms as T

class TransformsSimCLR:

    """
    Class which is used to transform the data before being fed to the SimCLR.
    """
    def __init__(self, size, pretrained=True, n_channel=3, validation=False) -> None:
        self.train_transforms = T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(size=size, scale=(0.25, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # Taking the means of the normal distributions of the 3 channels
            # since we are moving to grayscale
            T.Normalize(mean=np.mean([0.485, 0.456, 0.406]).repeat(n_channel),
                        std=np.sqrt(
                            (np.array([0.229, 0.224, 0.225])**2).sum()/9).repeat(n_channel)
                        ) if pretrained is True else T.Lambda(lambda x: x)
        ])

        self.validation_transforms = T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(size=size, scale=(1.0, 1.0)),
            # Taking the means of the normal distributions of the 3 channels
            # since we are moving to grayscale
            T.Normalize(mean=np.mean([0.485, 0.456, 0.406]).repeat(n_channel),
                        std=np.sqrt(
                            (np.array([0.229, 0.224, 0.225])**2).sum()/9).repeat(n_channel)
                        ) if pretrained is True else T.Lambda(lambda x: x)
        ])
        
        self.validation = validation

    def __call__(self, x):
        if not self.validation:
            return self.train_transforms(x), self.train_transforms(x)
        else:
            return self.validation_transforms(x)


def get_simclr_model(IS_PRETRAINED: bool, n_channel: int, embedding_size: int, device: str):
    """
    Combines the ResNet with the Pytorch implementation of the SimCLR
    :param IS_PRETRAINED: True if using a pretrained model
    :param n_channel: number of channels to use
    :param embedding_size: size of the embeddings in the model
    :param device: device on which the model is trained
    :return: torch model
    """
    resnet_encoder = torchvision.models.resnet50(pretrained=IS_PRETRAINED)

    ## Experimental setup for multiplying the grayscale channel
    ## https://stackoverflow.com/a/54777347
    weight = resnet_encoder.conv1.weight.clone()
    resnet_encoder.conv1 = torch.nn.Conv2d(n_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # normalize back by n_channels after tiling
    resnet_encoder.conv1.weight.data = weight.sum(dim=1, keepdim=True).tile(1, n_channel, 1, 1)/n_channel


    n_features = resnet_encoder.fc.in_features
    model = SimCLR(resnet_encoder, embedding_size, n_features)
    model = model.to(device)
    
    return model
