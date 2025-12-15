import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from string import ascii_uppercase, digits, punctuation

valid_chars = ascii_uppercase + digits + punctuation + " \t\n"


SPLIT_DIR = "./data/splits"

class SROIEDataset(Dataset):
    def __init__(self, split, image_dir, annotation_dir, target_size=(800, 600)):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.target_size = target_size  # (width, height)
        root=None
        if split == 'train':
            root = "../../data/splits/train_files.txt"
        elif split == 'val':
            root = "../../data/splits/val_files.txt"
        if root is not None:
            with open(root, 'r', encoding='utf-8') as f:
                mask_files = [line.strip() for line in f if line.strip()]
            self.image_files = sorted([ f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f in mask_files])
        else:
            self.image_files = sorted([ f for f in os.listdir(image_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ресайз с сохранением пропорций
        image, scale_w, scale_h = self._resize_with_padding(image)
        h, w = image.shape[:2]
        
        # Загрузка аннотаций
        base_name = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.annotation_dir, f"{base_name}.txt")
        
        boxes = []  # Конвертированные в горизонтальные прямоугольники
        original_quads = []  # Оригинальные квадроугольники
        texts = []

            # Пробуем разные кодировки
   
        content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                    with open(ann_path, 'r', encoding=encoding) as f:
                        for line_no, line in enumerate(f):
                            parts = line.strip().split(',', 8)
                            for c in parts[-1]:
                                if not c in valid_chars:
                                    print(f"Invalid char {repr(c)} in {f.name} on line {line_no}")
                            if len(parts) == 9:
                                # Оригинальный квадроугольник
                                quad = list(map(float, parts[:8]))
                                text = parts[8]
                                
                                # Конвертируем в горизонтальный прямоугольник и масштабируем
                                rect = self._quad_to_rect(quad)
                                rect[0] *= scale_w  # x_min
                                rect[1] *= scale_h  # y_min
                                rect[2] *= scale_w  # x_max
                                rect[3] *= scale_h  # y_max
                                
                                boxes.append(rect)
                                original_quads.append(quad)
                                texts.append(text)
            except UnicodeDecodeError:
                print(f'Ошибка с {encoding}')
                continue
   
 
        # Преобразование изображения в тензор
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image_tensor,  # [3, H, W]
            'boxes': boxes,  # Масштабированные горизонтальные прямоугольники
            'original_quads': original_quads,  # Оригинальные квадры
            'texts': texts,
            'image_id': base_name,
            'original_size': image.shape[:2],
            'scale_factors': (scale_w, scale_h)
        }
    
    def _resize_with_padding(self, image):
        """Ресайз с сохранением пропорций и паддингом"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Вычисляем масштаб
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Ресайз
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Паддинг
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized
        
        return padded, scale, scale
    
    def _quad_to_rect(self, quad):
        """Конвертация квадроугольника в горизонтальный прямоугольник"""
        x_coords = [quad[0], quad[2], quad[4], quad[6]]
        y_coords = [quad[1], quad[3], quad[5], quad[7]]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
