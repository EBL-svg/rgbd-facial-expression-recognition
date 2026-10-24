import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Attention par Canal
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # Attention Spatiale
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Canal
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(spatial_att))
        x = x * spatial_att
        return x

class RGBDEfficientNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(RGBDEfficientNet, self).__init__()
        
        # 1. Projection Layer (4 canaux -> 3 canaux)
        self.projection = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)
        self.relu_proj = nn.ReLU(inplace=True)

        # 2. Backbone EfficientNetV2-S (Pretrained)
        weights = (
        EfficientNet_V2_S_Weights.IMAGENET1K_V1
        if pretrained else None
        )
        self.base_model = efficientnet_v2_s(weights=weights)
        self.feature_extractor = self.base_model.features
        
        # 3. CBAM Block (Input features = 1280 pour EffNetV2S)
        self.cbam = CBAMBlock(channel=1280)

        # 4. Custom Head (identique à ton code TF : Conv256 -> GAP -> Dense128 -> Softmax)
        self.final_conv = nn.Conv2d(1280, 256, kernel_size=1)
        self.relu_final = nn.ReLU(inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes) 
            # Note: Softmax est géré par CrossEntropyLoss en PyTorch
        )

    def forward(self, x):
        # x: (Batch, 4, 224, 224)
        x = self.projection(x)
        x = self.relu_proj(x)
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = self.final_conv(x)
        x = self.relu_final(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x