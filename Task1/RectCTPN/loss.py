import torch
import torch.nn as nn
import torch.optim as optim
class CTPNLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_reg=1.0, lambda_side=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_side = lambda_side
        
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        self.side_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, cls_pred, reg_pred, side_pred, gt_data):
        cls_target = gt_data['cls_targets'].to(cls_pred.device).long()
        reg_target = gt_data['reg_targets'].to(reg_pred.device)
        side_target = gt_data['side_targets'].to(side_pred.device)
        
        B, C, H, W = cls_pred.shape
        K = C // 2
        
        # Reshape predictions
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, K, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, K, 2)
        side_pred = side_pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, K, 2)
        
        # Reshape targets
        cls_target = cls_target.view(B, H, W, K, 2)
        reg_target = reg_target.view(B, H, W, K, 2)  # Изменили с 3 на 2 канала
        side_target = side_target.view(B, H, W, K, 2)
        
        # Classification loss
        cls_pred_flat = cls_pred.view(-1, 2)
        cls_target_flat = cls_target.view(-1, 2).argmax(dim=1)
        cls_loss = self.cls_loss(cls_pred_flat, cls_target_flat)
        
        # Regression loss (only for positive anchors)
        pos_mask = (cls_target[..., 1] > 0.5).float().unsqueeze(-1)
        
        if pos_mask.sum() > 0:
            reg_loss = self.reg_loss(reg_pred * pos_mask, reg_target * pos_mask)
        else:
            reg_loss = torch.tensor(0.0, device=cls_pred.device)
        
        # Side-refinement loss
        if pos_mask.sum() > 0:
            side_loss = self.side_loss(side_pred * pos_mask, side_target * pos_mask)
        else:
            side_loss = torch.tensor(0.0, device=cls_pred.device)
        
        total_loss = (self.lambda_cls * cls_loss + 
                     self.lambda_reg * reg_loss + 
                     self.lambda_side * side_loss)
        
        return {
            'total': total_loss,
            'cls': cls_loss,
            'reg': reg_loss,
            'side': side_loss
        }