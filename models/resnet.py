import timm
import torch.nn as nn


class ResNet_50(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_dim = 256

        self.backbone = timm.create_model(
            'resnet50', pretrained=True, features_only=True, out_indices=[3], in_chans=1)

        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, self.final_dim)
        )

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.final_dim, 1)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        (B, C, D, H, W) = x.shape
        x = x.permute(0, 2, 1, 3, 4).view(B * D, C, H, W)
        x = self.backbone(x)[0]

        _C, _H, _W = x.shape[-3], x.shape[-2], x.shape[-1]
        x = self.pool(x).view(B * D, _C)
        x = self.linear(x)
        x = self.head(x)
        x = x.contiguous().view(B, D)
        x = x.mean(dim=1, keepdims=True)
        return x
