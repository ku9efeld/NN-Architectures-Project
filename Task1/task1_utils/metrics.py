import numpy as np
import cv2
from typing import List, Tuple

def polygon_to_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    """Преобразует 4-угольник (8 координат) в axis-aligned bounding box."""
    x_coords = poly[0::2]
    y_coords = poly[1::2]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    return x_min, y_min, x_max, y_max


def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    """Вычисляет IoU между двумя axis-aligned bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def compute_f1_for_image(
    pred_polys: List[np.ndarray],
    gt_polys: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Вычисляет TP, FP, FN для одного изображения.
    
    Args:
        pred_polys: список предсказанных полигонов (каждый — [x1,y1,...,x4,y4])
        gt_polys: список GT полигонов (того же формата)
        iou_threshold: порог IoU для совпадения
    
    Returns:
        (TP, FP, FN)
    """
    if len(gt_polys) == 0:
        return 0, len(pred_polys), 0
    if len(pred_polys) == 0:
        return 0, 0, len(gt_polys)
    
    if len(pred_polys[0]) == 8:
        # Преобразуем в AABB для IoU (стандарт в SROIE)
        pred_boxes = [polygon_to_bbox(p) for p in pred_polys]
    else:
        pred_boxes = pred_polys

    if len(gt_polys[0]) == 8:
        # Преобразуем в AABB для IoU (стандарт в SROIE)
        gt_boxes = [polygon_to_bbox(g) for g in gt_polys]
    else:
        gt_boxes = gt_polys

    matched_gt = set()
    tp = 0

    for pred_box in pred_boxes:
        best_iou = -1
        best_gt_idx = -1
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_polys) - tp
    fn = len(gt_polys) - tp
    return tp, fp, fn

def compute_f1_score(
    all_predictions: List[List[np.ndarray]],
    all_ground_truths: List[List[np.ndarray]],
    iou_threshold: float = 0.5
) -> float:
    """
    Вычисляет микро-F1 по всему датасету.
    
    Args:
        all_predictions: список по изображениям → список полигонов
        all_ground_truths: аналогично
    """
    total_tp = total_fp = total_fn = 0

    for preds, gts in zip(all_predictions, all_ground_truths):
        tp, fp, fn = compute_f1_for_image(preds, gts, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = total_tp / (total_tp + total_fp)

    if total_tp + total_fn == 0:
        recall = 0.0
    else:
        recall = total_tp / (total_tp + total_fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1