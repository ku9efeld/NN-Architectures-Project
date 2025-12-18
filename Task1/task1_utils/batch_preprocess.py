import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def ctpn_collate_fn(batch):
    """Collate функция для CTPN"""
    images = []
    boxes_list = []
    original_quads_list = []
    texts_list = []
    image_ids = []
    original_sizes = []
    scale_factors = []
    
    for item in batch:
        images.append(item['image'])
        boxes_list.append(torch.tensor(item['boxes'], dtype=torch.float32))
        original_quads_list.append(item['original_quads'])
        texts_list.append(item['texts'])
        image_ids.append(item['image_id'])
        original_sizes.append(item['original_size'])
        scale_factors.append(item['scale_factors'])
    
    # Стекируем изображения одинакового размера
    images = torch.stack(images, dim=0)
    
    return {
        'images': images,  # [B, 3, H, W]
        'boxes': boxes_list,  # список тензоров [N_i, 4]
        'original_quads': original_quads_list,
        'texts': texts_list,
        'image_ids': image_ids,
        'original_sizes': original_sizes,
        'scale_factors': scale_factors
    }


def prepare_full_ctpn_targets(batch, num_anchors=10): 
    """Таргеты для CTPN модели"""

    anchor_scales = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    
    num_anchors = len(anchor_scales)
    images = batch['images']
    boxes_list = batch['boxes']
    
    B, C, H, W = images.shape
    feature_stride = 16
    feat_h, feat_w = H // feature_stride, W // feature_stride  # 40, 40
    
    # ИЗМЕНЕНИЕ 1: Правильные размеры для CTPNLossFixed
    # Классификация: [B, H, W, K] (не one-hot!)
    cls_targets = torch.zeros(B, feat_h, feat_w, num_anchors, dtype=torch.float32)
    
    # Регрессия: [B, H, W, K, 2]
    reg_targets = torch.zeros(B, feat_h, feat_w, num_anchors, 2, dtype=torch.float32)
    
    # Side refinement: [B, H, W, K, 2]
    side_targets = torch.zeros(B, feat_h, feat_w, num_anchors, 2, dtype=torch.float32)
    
    for b_idx in range(B):
        boxes = boxes_list[b_idx]
        
        for box in boxes:
            if len(box) == 0:
                continue
            
            x_min, y_min, x_max, y_max = box
            gt_height = y_max - y_min
            gt_center_y = (y_min + y_max) / 2
            gt_center_x = (x_min + x_max) / 2
            
            feat_x = min(int(gt_center_x // feature_stride), feat_w - 1)
            feat_y = min(int(gt_center_y // feature_stride), feat_h - 1)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    curr_x = feat_x + dx
                    curr_y = feat_y + dy
                    
                    if not (0 <= curr_x < feat_w and 0 <= curr_y < feat_h):
                        continue
                    
                    for a_idx, anchor_h in enumerate(anchor_scales):
                        anchor_center_y = (curr_y + 0.5) * feature_stride
                        
                        anchor_y_min = anchor_center_y - anchor_h / 2
                        anchor_y_max = anchor_center_y + anchor_h / 2
                        
                        y_min_inter = max(anchor_y_min, y_min)
                        y_max_inter = min(anchor_y_max, y_max)
                        inter_h = max(0, y_max_inter - y_min_inter)
                        
                        union_h = (anchor_y_max - anchor_y_min) + (y_max - y_min) - inter_h
                        vertical_iou = inter_h / union_h if union_h > 0 else 0
                        
                        if vertical_iou > 0.7:
                            # ИЗМЕНЕНИЕ 2: cls_targets - просто 1 для positive
                            cls_targets[b_idx, curr_y, curr_x, a_idx] = 1.0
                            
                            dy = (gt_center_y - anchor_center_y) / anchor_h
                            dh = torch.log(torch.tensor(gt_height / anchor_h, dtype=torch.float32))
                            
                            reg_targets[b_idx, curr_y, curr_x, a_idx, 0] = dy
                            reg_targets[b_idx, curr_y, curr_x, a_idx, 1] = dh
                            
                            dx_left = (x_min - curr_x * feature_stride) / feature_stride
                            dx_right = (x_max - curr_x * feature_stride) / feature_stride
                            
                            side_targets[b_idx, curr_y, curr_x, a_idx, 0] = dx_left
                            side_targets[b_idx, curr_y, curr_x, a_idx, 1] = dx_right
                            
                        elif vertical_iou < 0.3:
                            # Только если еще не positive
                            if cls_targets[b_idx, curr_y, curr_x, a_idx] == 0:
                                cls_targets[b_idx, curr_y, curr_x, a_idx] = 0.0
    
    return {
        'cls_targets': cls_targets,      # [B, H, W, K]
        'reg_targets': reg_targets,      # [B, H, W, K, 2]
        'side_targets': side_targets     # [B, H, W, K, 2]
    }




def get_yolo_augmentations(img_size=640):
    """
    YOLO-style трансформации с letterbox padding
    Args:
        img_size: целевой размер изображения (квадрат)
    Returns:
        Albumentations композиция трансформаций
    """
    return A.Compose([
        # Масштабирование по длинной стороне с сохранением пропорций
        A.LongestMaxSize(max_size=img_size),
        
        # Добавление серых полей для получения квадратного изображения
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # Серый цвет как в YOLO
        ),
        
        # Нормализация (можно изменить на свои значения)
        A.Normalize(
            mean=[0, 0, 0],  # или [0.485, 0.456, 0.406] для ImageNet
            std=[1, 1, 1],   # или [0.229, 0.224, 0.225]
            max_pixel_value=255.0
        ),
        
        # Конвертация в тензор PyTorch
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',  # формат bbox: [x_min, y_min, x_max, y_max]
        label_fields=['class_labels']
    ))