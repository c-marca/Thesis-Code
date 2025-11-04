import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn
import torch.onnx                       # To export model to ONXX
import warnings  
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)
m = ptcv_get_model('mobilenet_wd4', pretrained=True)  # wd4 = width 0.25

'''
dummy_input = torch.randn(1, 3, 128, 128) 
if __name__ == "__main__":

    torch.onnx.export(
        m, 
        dummy_input, 
        "MobileNet_original.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
'''
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


MobileNetLP = FrozenBackboneMobileNet(m)
MobileNetLP.name = "MobileNetV1_025_Linear_Probe"
MobileNetFD = ptcv_get_model('fdmobilenet_wd4', pretrained=True)  # FD-MobileNet ×0.25


# replace classifier for 2 classes

MobileNetFD.output = nn.Linear(MobileNetFD.output.in_features, 2)
replace_avgpool_with_adapt(MobileNetFD)
MobileNetFD = FrozenBackboneMobileNet(MobileNetFD)
MobileNetFD.name = "MobileNetV1_025_Fast_Downsampling_Linear_Probe"
MobileNet_no_pre = ptcv_get_model('mobilenet_wd4', pretrained=False)
MobileNet_no_pre.output = nn.Linear(MobileNet_no_pre.output.in_features, 2)
MobileNet_no_pre.name = "MobileNetV1_025_no_pretraining"
replace_avgpool_with_adapt(MobileNet_no_pre)


m = ptcv_get_model('mobilenet_wd4', pretrained=True)  # wd4 = width 0.25
m.output = nn.Linear(m.output.in_features, 2)
replace_avgpool_with_adapt(m)
for p in m.parameters():
    p.requires_grad = False

# Unfreeze last K depthwise-separable blocks + new head
K = 2  # tune
tail_blocks = list(m.features.children())[-K:]
for blk in tail_blocks:
    for p in blk.parameters():
        p.requires_grad = True

for p in m.output.parameters():
    p.requires_grad = True

MobileNetFT = m
MobileNetFT.name = "MobileNetV1_025__Partial_FineTuning_K2"
