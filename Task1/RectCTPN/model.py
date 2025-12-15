import torch
import torch.nn as nn
import torchvision.models as models


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiLSTMBlock(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.conv = nn.Conv2d(hidden_dim * 2, 512, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape для LSTM: [B, C, H, W] -> [B*H, W, C]
        x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_reshaped = x_perm.reshape(B * H, W, C)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x_reshaped)  # [B*H, W, hidden*2]
        
        # Reshape обратно
        lstm_out = lstm_out.reshape(B, H, W, -1)
        lstm_out = lstm_out.permute(0, 3, 1, 2)  # [B, hidden*2, H, W]
        
        return self.conv(lstm_out)


class CTPN(nn.Module):
    def __init__(self, num_anchors=10):
        super().__init__()
        self.num_anchors = num_anchors
        
        # VGG16 backbone (первые 30 слоев до 5-го пулинга)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(vgg16.features.children())[:30])
        
        # RPN конволюции
        self.rpn_conv1 = BasicConv(512, 512)
        self.rpn_conv2 = BasicConv(512, 512)
        
        # BiLSTM для контекста
        self.bilstm = BiLSTMBlock(input_dim=512, hidden_dim=128)
        
        # Головы
        self.cls_head = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        self.reg_head = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        self.side_head = nn.Conv2d(512, num_anchors * 2, kernel_size=1)

    def forward(self, x):
        # Backbone features
        features = self.backbone(x)  # [B, 512, H', W']
        
        # RPN features
        x = self.rpn_conv1(features)
        x = self.rpn_conv2(x)
        
        # BiLSTM
        x = self.bilstm(x)
        
        # Predictions
        cls_pred = self.cls_head(x)  # [B, num_anchors*2, H', W']
        reg_pred = self.reg_head(x)   # [B, num_anchors*2, H', W']
        side_pred = self.side_head(x) # [B, num_anchors*2, H', W']
        
        return cls_pred, reg_pred, side_pred