
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn
import timm

m = ptcv_get_model('mobilenet_wd4', pretrained=True)  # wd4 = width 0.25
# replace head for your classes
m.output = nn.Linear(m.output.in_features, 2)

# replace avgpool with adaptivepool in order to support resolution changes from the original 224 resolution
def replace_avgpool_with_adapt(module):                             
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AvgPool2d):
            setattr(module, name, nn.AdaptiveAvgPool2d(1))
        else:
            replace_avgpool_with_adapt(child)

replace_avgpool_with_adapt(m)

# optional: freeze backbone

class FrozenBackboneMobileNet(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.backbone = base.features            # conv + pooling stack
        self.head = base.output                  # final Linear
        # freeze params in backbone and fix BN/dropout behavior
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)                 # no autograd, no activations kept
        x = x.view(x.size(0), -1)                # flatten after AdaptiveAvgPool2d(1)
        return self.head(x)                      # trainable head only
'''
class PartialFT_MobileNet(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.backbone = base.features            # conv + pooling stack
        self.head = base.output                  # final Linear
        # freeze params in backbone and fix BN/dropout behavior
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)                 # no autograd, no activations kept
        x = x.view(x.size(0), -1)                # flatten after AdaptiveAvgPool2d(1)
        return self.head(x)                      # trainable head only
'''

MobileNetLP = FrozenBackboneMobileNet(m)

MobileNetFD= ptcv_get_model('fdmobilenet_wd4', pretrained=True)  # FD-MobileNet ×0.25

# replace classifier for 2 classes
MobileNetFD.output = nn.Linear(m.output.in_features, 2)
replace_avgpool_with_adapt(MobileNetFD)
MobileNetFD = FrozenBackboneMobileNet(MobileNetFD)

m = ptcv_get_model('mobilenet_wd4', pretrained=False)  # MobileNetV1 not pretrained
m.output = nn.Linear(m.output.in_features, 2)
MobileNet_no_pt = replace_avgpool_with_adapt(m)
