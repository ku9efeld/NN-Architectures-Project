import torch
import torch.nn as nn
#from torchvision.models import vgg16
import torchvision.models as models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class CustomCTPN(nn.Module):
    """
        Полная архитектура CTPN с 3 выходными головками:
        1. Classification: text/non-text (per anchor)
        2. Vertical regression: (dy, dh) для каждого якоря
        3. Side-refinement: (dx_left, dx_right) для каждого якоря
        
        Args:
            backbone_layers: Количество слоев VGG16 для backbone
            num_anchors: Количество anchor scales (по умолчанию 10 как в оригинале)
    """
    def __init__(self, backbone_layers=5, num_anchors=10):
        super(CustomCTPN, self).__init__()
        self.num_anchors = num_anchors
        
        # Загружаем VGG16
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg16.features.children())[:backbone_layers]
        self.backbone = nn.Sequential(*features)
        
        # Определяем реальный stride
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            out = self.backbone(dummy)
            self.backbone_channels = out.shape[1] # 64
            # Вычисляем реальный stride
            self.backbone_stride = 640 // out.shape[2] #2
        
        print(f"Backbone output: {self.backbone_channels} channels, stride: {self.backbone_stride}")
        
          # Конфигурация для stride=2
        intermediate_channels = 128  # Уменьшаем для stride=2
        rnn_hidden = 64
        fc_out_channels = 128
        
        # Intermediate layers
        self.conv1 = nn.Conv2d(self.backbone_channels, intermediate_channels, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # BiLSTM
        self.rnn = nn.LSTM(
            input_size=intermediate_channels,
            hidden_size=rnn_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # FC layer
        self.fc = nn.Linear(rnn_hidden * 2, fc_out_channels)
        self.relu_fc = nn.ReLU(inplace=True)
        
        # Heads
        self.cls_head = nn.Conv2d(fc_out_channels, num_anchors * 2, 1)
        self.reg_head = nn.Conv2d(fc_out_channels, num_anchors * 2, 1)
        self.side_head = nn.Conv2d(fc_out_channels, num_anchors * 2, 1)
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 64, 320, 320]
        
        x = self.relu1(self.conv1(features))  # [B, 128, 320, 320]
        x = self.relu2(self.conv2(x))         # [B, 128, 320, 320]
        
        B, C, H, W = x.shape
        
        # BiLSTM
        x_rnn = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        rnn_out, _ = self.rnn(x_rnn)  # [B*320, 320, 128]
        rnn_out = rnn_out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # FC
        rnn_out_perm = rnn_out.permute(0, 2, 3, 1)
        fc_out = self.relu_fc(self.fc(rnn_out_perm))
        fc_out = fc_out.permute(0, 3, 1, 2).contiguous()  # [B, 128, 320, 320]
        
        # Heads
        cls_out = self.cls_head(fc_out)  # [B, 20, 320, 320]
        reg_out = self.reg_head(fc_out)
        side_out = self.side_head(fc_out)
        
        return cls_out, reg_out, side_out