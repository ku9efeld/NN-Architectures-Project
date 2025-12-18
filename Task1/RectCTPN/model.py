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
    



class CTPNAnchors:
    def __init__(self, anchor_heights=None, stride=16, width=16):
        self.stride = stride
        self.width = width
        
        if anchor_heights is None:
            self.anchor_heights = [4, 7, 11, 18, 28, 44, 70, 110, 170, 260]
        else:
            self.anchor_heights = anchor_heights
        
        self.num_anchors = len(self.anchor_heights)
        self.base_anchors = self._generate_base_anchors()
    
    def _generate_base_anchors(self):
        """Базовые anchors относительно центра (0,0)."""
        anchors = []
        for h in self.anchor_heights:
            anchors.append([
                -self.width / 2,  # x1
                -h / 2,           # y1
                self.width / 2,   # x2
                h / 2             # y2
            ])
        return torch.tensor(anchors, dtype=torch.float32)
    
    def generate_all_anchors(self, feature_shape, device='cpu'):
        """
        Генерация всех anchors для feature map.
        
        Args:
            feature_shape: (H, W) размер feature map
            device: 'cpu' или 'cuda'
        
        Returns:
            anchors: [H, W, num_anchors, 4]
        """
        H, W = feature_shape
        
        # Сетка центров на feature map
        shift_x = torch.arange(0, W, device=device).float()
        shift_y = torch.arange(0, H, device=device).float()
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        # Центры в пикселях входного изображения
        centers_x = (shift_x + 0.5) * self.stride  # [H, W]
        centers_y = (shift_y + 0.5) * self.stride  # [H, W]
        
        centers = torch.stack([centers_x, centers_y], dim=-1)  # [H, W, 2]
        
        # Базовые anchors
        base_anchors = self.base_anchors.to(device)  # [num_anchors, 4]
        
        # Расширяем до [H, W, num_anchors, 4]
        anchors = base_anchors.view(1, 1, self.num_anchors, 4).expand(H, W, -1, -1)
        
        # Сдвигаем
        anchors[..., 0::2] += centers[..., 0:1]  # x1, x2 + center_x
        anchors[..., 1::2] += centers[..., 1:2]  # y1, y2 + center_y
        
        return anchors
    
    def anchor_targets(self, gt_boxes, feature_shape, img_shape):
        """
        Создает таргеты для обучения.
        """
        H, W = feature_shape
        anchors = self.generate_all_anchors(feature_shape)  # [H, W, K, 4]
        
        # Инициализируем таргеты
        cls_targets = torch.zeros(H, W, self.num_anchors, dtype=torch.float32)
        reg_targets = torch.zeros(H, W, self.num_anchors, 2, dtype=torch.float32)
        side_targets = torch.zeros(H, W, self.num_anchors, 2, dtype=torch.float32)
        
        for gt_box in gt_boxes:
            if len(gt_box) < 4:
                continue
                
            x1_gt, y1_gt, x2_gt, y2_gt = gt_box
            gt_height = y2_gt - y1_gt
            gt_center_y = (y1_gt + y2_gt) / 2
            gt_center_x = (x1_gt + x2_gt) / 2
            
            feat_x = int(torch.clamp(gt_center_x // self.stride, 0, W - 1))
            feat_y = int(torch.clamp(gt_center_y // self.stride, 0, H - 1))
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    curr_x = feat_x + dx
                    curr_y = feat_y + dy
                    
                    if not (0 <= curr_x < W and 0 <= curr_y < H):
                        continue
                    
                    for a_idx in range(self.num_anchors):
                        anchor = anchors[curr_y, curr_x, a_idx]
                        x1_a, y1_a, x2_a, y2_a = anchor
                        
                        # Vertical IoU
                        y_min_inter = max(y1_a, y1_gt)
                        y_max_inter = min(y2_a, y2_gt)
                        inter_h = max(0, y_max_inter - y_min_inter)
                        
                        union_h = (y2_a - y1_a) + (y2_gt - y1_gt) - inter_h
                        
                        if union_h <= 0:
                            continue
                            
                        v_iou = inter_h / union_h
                        
                        if v_iou > 0.7:
                            cls_targets[curr_y, curr_x, a_idx] = 1.0
                            
                            anchor_center_y = (y1_a + y2_a) / 2
                            anchor_height = y2_a - y1_a
                            
                            dy_val = (gt_center_y - anchor_center_y) / anchor_height
                            dh_val = torch.log(gt_height / anchor_height)
                            
                            reg_targets[curr_y, curr_x, a_idx, 0] = dy_val
                            reg_targets[curr_y, curr_x, a_idx, 1] = dh_val
                            
                            dx_left = (x1_gt - curr_x * self.stride) / self.stride
                            dx_right = (x2_gt - curr_x * self.stride) / self.stride
                            
                            side_targets[curr_y, curr_x, a_idx, 0] = dx_left
                            side_targets[curr_y, curr_x, a_idx, 1] = dx_right
                        
                        elif v_iou < 0.3:
                            if cls_targets[curr_y, curr_x, a_idx] == 0:
                                cls_targets[curr_y, curr_x, a_idx] = 0.0
        
        return {
            'cls_targets': cls_targets,
            'reg_targets': reg_targets,
            'side_targets': side_targets
        }