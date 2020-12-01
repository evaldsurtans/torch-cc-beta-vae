import torch
import torchvision.models

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = torchvision.models.resnet18()
        #TODO

    def forward(self, x):
        #TODO
        pass

