import torch
import torchvision
from torchvision import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18()
        self.fc = nn.Linear(self.resnet18.fc.in_features, 3)
        self.resnet18.fc = self.fc

    def forward(self, x):
        x = self.resnet18(x)
        return x
