import torch
import torch.nn as nn
import torch.optim as optim
class CTPNLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_reg=1.0, lambda_side=1.0):
         super().__init__()
         self.lambda_cls = lambda_cls
         self.lambda_reg = lambda_reg
         self.lambda_side = lambda_side
        
        # Вариант 1: BCEWithLogitsLoss для бинарной классификации
         self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Вариант 2: CrossEntropyLoss с индексами (если нужно)
        # self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
        
         self.reg_loss = nn.SmoothL1Loss(reduction='sum')  # sum, потом поделим на num_pos
         self.side_loss = nn.SmoothL1Loss(reduction='sum')
    
    def forward(self, cls_pred, reg_pred, side_pred, gt_data):
        # Предполагаем, что gt_data содержит:
        # cls_target: [B, H, W, K] с 0 или 1 (binary)
        # reg_target: [B, H, W, K, 2] 
        # side_target: [B, H, W, K, 2]
        
        cls_target = gt_data['cls_targets'].float().to(cls_pred.device)  # [B, H, W, K]
        reg_target = gt_data['reg_targets'].to(reg_pred.device)          # [B, H, W, K, 2]
        side_target = gt_data['side_targets'].to(side_pred.device)       # [B, H, W, K, 2]
        
        B, C, H, W = cls_pred.shape
        K = C // 2  # 10 anchors
        
        # Reshape predictions в удобный формат
        # cls_pred: [B, 2*K, H, W] -> [B, H, W, K, 2]
        cls_pred = cls_pred.view(B, 2, K, H, W).permute(0, 3, 4, 2, 1)
        
        # reg_pred: [B, 2*K, H, W] -> [B, H, W, K, 2]
        reg_pred = reg_pred.view(B, 2, K, H, W).permute(0, 3, 4, 2, 1)
        
        # side_pred: [B, 2*K, H, W] -> [B, H, W, K, 2]
        side_pred = side_pred.view(B, 2, K, H, W).permute(0, 3, 4, 2, 1)
        
        # ===== 1. CLASSIFICATION LOSS =====
        # Вариант A: BCEWithLogitsLoss (рекомендуется)
        # Берем только positive channel для бинарной классификации
        cls_pred_pos = cls_pred[..., 1]  # [B, H, W, K] - логиты для класса "текст"
        cls_loss = self.cls_loss(cls_pred_pos, cls_target)
        
        # ИЛИ Вариант B: CrossEntropyLoss
        # cls_pred_flat = cls_pred.view(-1, 2)  # [B*H*W*K, 2]
        # cls_target_flat = cls_target.view(-1).long()  # [B*H*W*K] с 0 или 1
        # cls_loss = self.cls_loss(cls_pred_flat, cls_target_flat)
        
        # ===== 2. POSITIVE MASK =====
        pos_mask = (cls_target > 0.5)  # [B, H, W, K], bool
        num_pos = pos_mask.sum().float().clamp(min=1.0)
        
        # ===== 3. REGRESSION LOSS =====
        if num_pos > 0:
            # Расширяем маску для 2 каналов регрессии
            pos_mask_expanded = pos_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 2)  # [B, H, W, K, 2]
            
            # Выбираем только positive anchors
            reg_pred_pos = reg_pred[pos_mask_expanded].view(-1, 2)      # [num_pos*2,]
            reg_target_pos = reg_target[pos_mask_expanded].view(-1, 2)  # [num_pos*2,]
            
            reg_loss = self.reg_loss(reg_pred_pos, reg_target_pos) / num_pos
        else:
            reg_loss = torch.tensor(0.0, device=cls_pred.device)
        
        # ===== 4. SIDE REFINEMENT LOSS =====
        if num_pos > 0:
            # Используем ту же маску
            side_pred_pos = side_pred[pos_mask_expanded].view(-1, 2)      # [num_pos*2,]
            side_target_pos = side_target[pos_mask_expanded].view(-1, 2)  # [num_pos*2,]
            
            side_loss = self.side_loss(side_pred_pos, side_target_pos) / num_pos
        else:
            side_loss = torch.tensor(0.0, device=cls_pred.device)
        
        # ===== 5. TOTAL LOSS =====
        total_loss = (self.lambda_cls * cls_loss + 
                     self.lambda_reg * reg_loss + 
                     self.lambda_side * side_loss)
    
        return {
            'total': total_loss,
            'cls': cls_loss,
            'reg': reg_loss,
            'side': side_loss
        }
    

class CTPNLossSimple(nn.Module):
    """Простой loss для формата [B, 20, H, W]"""
    def __init__(self, lambda_cls=1.0, lambda_reg=1.0, lambda_side=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_side = lambda_side
        
        # Для формата [B, 20, H, W] где 20 = 10 anchors × 2
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
        self.reg_loss = nn.SmoothL1Loss(reduction='sum')
        self.side_loss = nn.SmoothL1Loss(reduction='sum')
    
    def forward(self, cls_pred, reg_pred, side_pred, gt_data):
        # gt_data: словарь с тензорами
        cls_target = gt_data['cls_targets']  # [B, H, W, K]
        reg_target = gt_data['reg_targets']  # [B, H, W, K, 2]
        side_target = gt_data['side_targets']  # [B, H, W, K, 2]
        
        B, C, H, W = cls_pred.shape  # [B, 20, H, W]
        num_anchors = C // 2  # 10
        
        # 1. Reshape cls_pred для CrossEntropyLoss
        # cls_pred: [B, 20, H, W] -> [B, 2, 10, H, W]
        cls_pred_reshaped = cls_pred.view(B, 2, num_anchors, H, W)
        
        # 2. Reshape cls_target для CrossEntropyLoss
        # cls_target: [B, H, W, K] -> [B, K, H, W]
        cls_target_permuted = cls_target.permute(0, 3, 1, 2)  # [B, 10, H, W]
        
        # 3. Преобразуем в one-hot для CrossEntropy
        # Создаем отрицательный канал (background)
        cls_target_bg = (cls_target_permuted == 0).float()  # [B, 10, H, W]
        cls_target_fg = (cls_target_permuted == 1).float()  # [B, 10, H, W]
        
        # Объединяем в [B, 2, 10, H, W]
        cls_target_reshaped = torch.stack([cls_target_bg, cls_target_fg], dim=1)
        
        # 4. Classification loss
        cls_pred_flat = cls_pred_reshaped.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2)
        cls_target_flat = cls_target_reshaped.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2).argmax(dim=1)
        cls_loss = self.cls_loss(cls_pred_flat, cls_target_flat)
        
        # 5. Positive mask
        pos_mask = (cls_target_permuted > 0.5).float()  # [B, 10, H, W]
        num_pos = pos_mask.sum().clamp(min=1.0)
        
        # 6. Reshape reg pred/target
        reg_pred_reshaped = reg_pred.view(B, 2, num_anchors, H, W)  # [B, 2, 10, H, W]
        side_pred_reshaped = side_pred.view(B, 2, num_anchors, H, W)  # [B, 2, 10, H, W]
        
        # 7. Regression loss
        if num_pos > 0:
            # Расширяем маску для 2 каналов
            pos_mask_expanded = pos_mask.unsqueeze(1)  # [B, 1, 10, H, W]
            
            # Выбираем только positive anchors
            reg_pred_pos = reg_pred_reshaped[pos_mask_expanded.expand_as(reg_pred_reshaped) > 0.5].view(-1)
            reg_target_pos = reg_target.view(B, 2, num_anchors, H, W)[pos_mask_expanded.expand_as(reg_pred_reshaped) > 0.5].view(-1)
            
            side_pred_pos = side_pred_reshaped[pos_mask_expanded.expand_as(side_pred_reshaped) > 0.5].view(-1)
            side_target_pos = side_target.view(B, 2, num_anchors, H, W)[pos_mask_expanded.expand_as(side_pred_reshaped) > 0.5].view(-1)
            
            reg_loss = self.reg_loss(reg_pred_pos, reg_target_pos) / num_pos
            side_loss = self.side_loss(side_pred_pos, side_target_pos) / num_pos
        else:
            reg_loss = torch.tensor(0.0, device=cls_pred.device)
            side_loss = torch.tensor(0.0, device=cls_pred.device)
        
        total_loss = (self.lambda_cls * cls_loss + 
                     self.lambda_reg * reg_loss + 
                     self.lambda_side * side_loss)
        
        return {
            'total': total_loss,
            'cls': cls_loss,
            'reg': reg_loss,
            'side': side_loss,
            'num_pos': num_pos.item()
        }