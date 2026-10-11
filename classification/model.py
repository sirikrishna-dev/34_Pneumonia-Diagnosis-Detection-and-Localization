"""
10-layer CNN that is compatible with Class Activation Maps (CAM).

Rules from the paper that this follows:
  - 10 conv layers, 3x3 kernels, zero padding ("same"), ReLU
  - ONE Global Average Pooling layer, then ONE fully connected layer -> 2 classes
  - Part 2 (CAM) uses  model.features(x)  and  model.fc.weight[pneumonia_class]

NOTE: this file is shared with Person B (localization). Do not change the
class/method names without telling them.
"""
import torch
import torch.nn as nn

# (out_channels) per conv layer; "M" = 2x2 max pool.  10 convs in total.
CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 256, 256, 256]


class CAMNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, use_bn=True):
        super().__init__()
        layers, c = [], in_channels
        for v in CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.Conv2d(c, v, kernel_size=3, padding=1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                c = v
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, num_classes)

    def features(self, x):
        """Output of the last conv block: (B, 256, H/8, W/8). Used for CAM."""
        return self.conv(x)

    def forward(self, x):
        f = self.features(x)
        return self.fc(self.gap(f).flatten(1))


if __name__ == "__main__":
    m = CAMNet(in_channels=1)
    x = torch.randn(2, 1, 128, 128)
    print("logits:", m(x).shape, "| features:", m.features(x).shape)
    print("conv layers:", sum(isinstance(l, nn.Conv2d) for l in m.conv))
