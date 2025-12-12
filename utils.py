import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple
import random

def visualize_batch_samples(
    batch: Dict[str, torch.Tensor],
    num_samples: int = 4,
    figsize: Tuple[int, int] = (15, 10),
    show_text: bool = True
) -> None:
    """
    Быстрая визуализация сэмплов из батча.
    
    Args:
        batch: Батч данных от DataLoader
        num_samples: Количество сэмплов для отображения
        figsize: Размер фигуры
        save_path: Путь для сохранения изображения
        show_text: Показывать ли текст из боксов
    """
    images = batch['images']  # [B, C, H, W]
    boxes = batch.get('boxes', None)  # [B, max_boxes, 4]
    texts = batch.get('texts', None)  # [B, max_boxes, max_len]
    masks = batch.get('masks', None)  # [B, max_boxes]
    
    batch_size = images.shape[0]
    num_samples = min(num_samples, batch_size)
    
    # Создаем сетку для отображения
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Случайные индексы для отображения
    indices = random.sample(range(batch_size), num_samples) if batch_size > num_samples else list(range(batch_size))
    
    for idx, ax_idx in zip(indices, range(num_samples)):
        ax = axes[ax_idx]
        
        # Получаем изображение
        img = images[idx].cpu().numpy()
        img = img.transpose(1, 2, 0)  # [H, W, C]
        
        # Денормализация если нужно
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Отображаем изображение
        ax.imshow(img)
        ax.axis('off')
        
        # Добавляем bounding boxes если есть
        if boxes is not None and masks is not None:
            img_h, img_w = img.shape[:2]
            current_boxes = boxes[idx].cpu().numpy()
            current_mask = masks[idx].cpu().numpy()
            
            # Отображаем только валидные боксы
            valid_boxes = current_boxes[current_mask]
            
            for box_idx, box in enumerate(valid_boxes):
                if box_idx >= 10:  # Ограничиваем количество боксов для читаемости
                    break
                    
                # Денормализуем координаты
                x_min = box[0] * img_w
                y_min = box[1] * img_h
                x_max = box[2] * img_w
                y_max = box[3] * img_h
                
                # Рисуем прямоугольник
                rect = plt.Rectangle(
                    (x_min, y_min), 
                    x_max - x_min, 
                    y_max - y_min,
                    linewidth=1, 
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Добавляем текст если есть
                if show_text and texts is not None:
                    current_texts = texts[idx].cpu().numpy()
                    if current_mask[box_idx] and box_idx < len(current_texts):
                        # Декодируем текст из ASCII
                        text_codes = current_texts[box_idx]
                        text = ''.join([chr(c) for c in text_codes if c > 0])
                        if text:
                            # Показываем только первые 20 символов
                            display_text = text[:20] + '...' if len(text) > 20 else text
                            ax.text(
                                x_min, y_min - 5, 
                                display_text,
                                fontsize=8, 
                                color='yellow',
                                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
                            )
        
        # Добавляем заголовок с информацией
        title_parts = []
        if 'image_ids' in batch:
            title_parts.append(f"ID: {batch['image_ids'][idx]}")
        
        if 'num_boxes' in batch:
            title_parts.append(f"Boxes: {batch['num_boxes'][idx].item()}")
        
        ax.set_title('\n'.join(title_parts), fontsize=10)
    
    # Скрываем пустые subplots
    for ax in axes[num_samples:]:
        ax.axis('off')
        ax.set_visible(False)
    
    plt.suptitle(f"Batch Samples (batch size: {batch_size})", fontsize=14, y=1.02)
    plt.tight_layout()
    
  