import pretrainedmodels

import torch.nn as nn
import torch.nn.functional as F


class FaceKeypointResNet34(nn.Module):
    def __init__(self, requires_grad):
        super(FaceKeypointResNet34, self).__init__()
        self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layer
        self.l0 = nn.Linear(512, 136)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0
