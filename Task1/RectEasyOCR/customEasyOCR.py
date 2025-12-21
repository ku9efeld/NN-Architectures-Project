import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch

class TextDetectionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(TextDetectionModel, self).__init__()

        # Use MobileNetV3 as backbone
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small()

        # Extract feature layers
        self.features = backbone.features

        # Get the output channels from features
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            features = self.features(dummy)
            self.feature_channels = features.shape[1]

        # Detection head (for score map)
        self.detection_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Score between 0-1
        )

    def forward(self, x):
        # Extract features
        features = self.features(x)

        # Generate score map
        score_map = self.detection_head(features)

        # Upsample to match input size
        score_map = F.interpolate(score_map, size=x.shape[2:], mode='bilinear', align_corners=False)

        return score_map