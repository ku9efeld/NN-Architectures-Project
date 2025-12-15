import torch

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
    # Anchor высоты для оригинального CTPN (stride=16)
    anchor_scales = [11, 16, 22, 32, 46, 66, 93, 134, 191, 273]  # Оригинальные высоты
    
    images = batch['images']
    boxes_list = batch['boxes']
    
    B, C, H, W = images.shape  # [B, 3, 640, 640]
    feature_stride = 16  # CTPN модель: 640/40 = 16 (40x40 feature map)
    feat_h, feat_w = H // feature_stride, W // feature_stride  # 40x40

    
    # Формат [B, C, H, W] - 10 anchors * 2 = 20 каналов
    cls_targets = torch.zeros(B, num_anchors * 2, feat_h, feat_w, dtype=torch.float32)
    reg_targets = torch.zeros(B, num_anchors * 2, feat_h, feat_w, dtype=torch.float32)
    side_targets = torch.zeros(B, num_anchors * 2, feat_h, feat_w, dtype=torch.float32)
    
    for b_idx in range(B):
        boxes = boxes_list[b_idx]
        
        for box in boxes:
            if len(box) == 0:
                continue
            
            x_min, y_min, x_max, y_max = box
            gt_height = y_max - y_min
            gt_center_y = (y_min + y_max) / 2
            gt_center_x = (x_min + x_max) / 2
            
            # Координаты на feature map
            feat_x = min(int(gt_center_x // feature_stride), feat_w - 1)
            feat_y = min(int(gt_center_y // feature_stride), feat_h - 1)
            
            # Проверяем область вокруг
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    curr_x = feat_x + dx
                    curr_y = feat_y + dy
                    
                    if not (0 <= curr_x < feat_w and 0 <= curr_y < feat_h):
                        continue
                    
                    for a_idx, anchor_h in enumerate(anchor_scales):
                        anchor_center_y = (curr_y + 0.5) * feature_stride
                        
                        # Vertical IoU
                        anchor_y_min = anchor_center_y - anchor_h / 2
                        anchor_y_max = anchor_center_y + anchor_h / 2
                        
                        y_min_inter = max(anchor_y_min, y_min)
                        y_max_inter = min(anchor_y_max, y_max)
                        inter_h = max(0, y_max_inter - y_min_inter)
                        
                        union_h = (anchor_y_max - anchor_y_min) + (y_max - y_min) - inter_h
                        vertical_iou = inter_h / union_h if union_h > 0 else 0
                        
                        cls_pos_idx = a_idx * 2 + 1
                        cls_neg_idx = a_idx * 2
                        reg_dy_idx = a_idx * 2
                        reg_dh_idx = a_idx * 2 + 1
                        
                        if vertical_iou > 0.7:
                            if cls_targets[b_idx, cls_pos_idx, curr_y, curr_x] == 0:
                                cls_targets[b_idx, cls_neg_idx, curr_y, curr_x] = 0.0
                                cls_targets[b_idx, cls_pos_idx, curr_y, curr_x] = 1.0
                                
                                dy = (gt_center_y - anchor_center_y) / anchor_h
                                dh = torch.log(gt_height / anchor_h)
                                
                                reg_targets[b_idx, reg_dy_idx, curr_y, curr_x] = dy
                                reg_targets[b_idx, reg_dh_idx, curr_y, curr_x] = dh
                                
                                dx_left = (x_min - curr_x * feature_stride) / feature_stride
                                dx_right = (x_max - curr_x * feature_stride) / feature_stride
                                
                                side_targets[b_idx, reg_dy_idx, curr_y, curr_x] = dx_left
                                side_targets[b_idx, reg_dh_idx, curr_y, curr_x] = dx_right        
                        elif vertical_iou < 0.3:
                            if cls_targets[b_idx, cls_pos_idx, curr_y, curr_x] == 0:
                                cls_targets[b_idx, cls_neg_idx, curr_y, curr_x] = 1.0
                                cls_targets[b_idx, cls_pos_idx, curr_y, curr_x] = 0.0
    
    return {
        'cls_targets': cls_targets,
        'reg_targets': reg_targets,
        'side_targets': side_targets
    }
