import torch
import torch.nn as nn
from torchvision.models import resnet18

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.model = resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 8)

    def forward(self, x):
        return self.model(x)
