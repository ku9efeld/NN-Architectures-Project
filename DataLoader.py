import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')
class SROIEDataLoader(Dataset):
    """
    DataLoader для SROIE с фиксированным размером выходов
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (512, 512),
        max_boxes: int = 50,        # Максимальное количество боксов
        max_text_len: int = 20      # Максимальная длина текста в боксе
    ):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.max_boxes = max_boxes
        self.max_text_len = max_text_len
        
        # Пути
        self.img_dir = os.path.join(data_root, split, 'img')
        self.box_dir = os.path.join(data_root, split, 'box')
        self.entities_dir = os.path.join(data_root, split, 'entities')
        
        # Получаем список изображений
        self.image_files = sorted([
            f for f in os.listdir(self.img_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        print(f"✅ Загружено {len(self)} изображений из {split}")
        print(f"   Макс боксов: {max_boxes}, Макс длина текста: {max_text_len}")
    
    def __len__(self):
        return len(self.image_files)
    
    def _parse_box_line(self, line: str):
        """Парсинг строки формата: x1,y1,x2,y2,x3,y3,x4,y4,текст"""
        parts = line.strip().split(',')
        
        if len(parts) < 8:  # Минимум 8 координат
            return None, ""
        
        try:
            # Первые 8 значений - координаты
            coords = list(map(float, parts[:8]))
            
            # Остальное - текст (объединяем если есть запятые в тексте)
            text = ','.join(parts[8:]) if len(parts) > 8 else ""
            
            return coords, text[:self.max_text_len]  # Обрезаем текст
        except ValueError:
            return None, ""
    
    def _load_boxes_and_texts(self, base_name: str, img_w: int, img_h: int):
        """Загрузка bounding boxes и текстов"""
        box_path = os.path.join(self.box_dir, f"{base_name}.txt")
        
        # Инициализируем с фиксированным размером
        boxes_tensor = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
        texts_tensor = torch.zeros((self.max_boxes, self.max_text_len), dtype=torch.long)
        mask_tensor = torch.zeros(self.max_boxes, dtype=torch.bool)
        
        if not os.path.exists(box_path):
            return boxes_tensor, texts_tensor, mask_tensor, 0
        
        with open(box_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        valid_count = 0
        for i, line in enumerate(lines):
            if valid_count >= self.max_boxes:
                break
                
            coords, text = self._parse_box_line(line)
            if coords is not None:
                # Преобразуем координаты
                polygon = np.array(coords, dtype=np.float32).reshape(4, 2)
                
                # Получаем bounding box
                x_min, y_min = polygon.min(axis=0)
                x_max, y_max = polygon.max(axis=0)
                
                # Нормализуем координаты
                x_min_norm = x_min / img_w
                y_min_norm = y_min / img_h
                x_max_norm = x_max / img_w
                y_max_norm = y_max / img_h
                
                # Сохраняем бокс
                boxes_tensor[valid_count] = torch.tensor([
                    x_min_norm, y_min_norm, x_max_norm, y_max_norm
                ])
                
                # Сохраняем текст (кодируем ASCII кодами)
                text_encoded = [ord(c) for c in text[:self.max_text_len]]
                if len(text_encoded) < self.max_text_len:
                    text_encoded += [0] * (self.max_text_len - len(text_encoded))
                
                texts_tensor[valid_count] = torch.tensor(text_encoded[:self.max_text_len])
                mask_tensor[valid_count] = True
                valid_count += 1
        
        return boxes_tensor, texts_tensor, mask_tensor, valid_count
    
    def _load_entities(self, base_name: str):
        """Загрузка структурированных данных"""
        entities_path = os.path.join(self.entities_dir, f"{base_name}.txt")
        entities = {
            'company': '',
            'date': '',
            'address': '',
            'total': ''
        }
        
        if os.path.exists(entities_path):
            with open(entities_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Парсинг полей
            for line in content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    if key in entities:
                        entities[key] = value.strip()
        
        return entities
    
    def __getitem__(self, idx):
        """Получение одного элемента - ВСЕГДА одинаковый размер!"""
        # Имя файла
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # Загрузка изображения
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        
        # Ресайз изображения
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0  # Нормализация
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        
        # Загрузка bounding boxes и текстов
        boxes, texts, mask, num_boxes = self._load_boxes_and_texts(
            base_name, original_w, original_h
        )
        
        # Загрузка entities
        entities = self._load_entities(base_name)
        
        # Векторизуем entities для одинакового размера
        # entities_vector = torch.tensor([
        #     hash(entities['company']) % 10000,
        #     hash(entities['date']) % 10000,
        #     hash(entities['address']) % 10000,
        #     float(entities['total']) if entities['total'].replace('.', '').isdigit() else 0.0
        # ], dtype=torch.float32)
        
        return {
            'images': img_tensor,           # [3, H, W]
            'boxes': boxes,                # [max_boxes, 4]
            'texts': texts,                # [max_boxes, max_text_len]
            'mask': mask,                  # [max_boxes]
            'num_boxes': num_boxes,        # scalar
            'entities': entities,   # [4]
            'image_id': base_name,
            'original_size': torch.tensor([original_h, original_w])
        }
