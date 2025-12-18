import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from string import ascii_uppercase, digits, punctuation

from Task1.task1_utils.ctpn_utils import cal_rpn

valid_chars = ascii_uppercase + digits + punctuation + " \t\n"

class SROIEDataset(Dataset):
    def __init__(self, split, image_dir, annotation_dir, target_size=(800, 600), 
                 transform=None):
        """
        Args:
            split: 'train' или 'val'
            image_dir: путь к изображениям
            annotation_dir: путь к аннотациям
            target_size: оригинальный целевой размер (width, height)
            transform: Albumentations трансформации (если None, используется только ресайз)
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.target_size = target_size  # (width, height)
        self.transform = transform
        
        # Определяем файл со списком изображений для split
        root = None
        if split == 'train':
            root = "../../data/splits/train_files.txt"
        elif split == 'val':
            root = "../../data/splits/val_files.txt"
        
        if root is not None and os.path.exists(root):
            with open(root, 'r', encoding='utf-8') as f:
                mask_files = [line.strip() for line in f if line.strip()]
            self.image_files = sorted([
                f for f in os.listdir(image_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f in mask_files
            ])
        else:
            self.image_files = sorted([
                f for f in os.listdir(image_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Загрузка аннотаций
        base_name = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.annotation_dir, f"{base_name}.txt")
        
        boxes = []  # Конвертированные в горизонтальные прямоугольники [x_min, y_min, x_max, y_max]
        original_quads = []  # Оригинальные квадроугольники
        texts = []
        class_labels = []  # Для Albumentations (все классы = 0)
        
        # Читаем аннотации
        if os.path.exists(ann_path):
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(ann_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    content = lines
                    break
                except UnicodeDecodeError:
                    continue
            
            if content:
                for line_no, line in enumerate(content):
                    parts = line.strip().split(',', 8)
                    if len(parts) == 9:
                        # Оригинальный квадроугольник [x1, y1, x2, y2, x3, y3, x4, y4]
                        quad = list(map(float, parts[:8]))
                        text = parts[8]
                        
                        # Конвертируем в горизонтальный прямоугольник
                        x_coords = [quad[0], quad[2], quad[4], quad[6]]
                        y_coords = [quad[1], quad[3], quad[5], quad[7]]
                        rect = [
                            min(x_coords),  # x_min
                            min(y_coords),  # y_min
                            max(x_coords),  # x_max
                            max(y_coords)   # y_max
                        ]
                        
                        boxes.append(rect)
                        original_quads.append(quad)
                        texts.append(text)
                        class_labels.append(0)  # Все объекты одного класса

        resized_image, scaled_boxes, scale_factors = self._simple_ctpn_transform(image, boxes)
        image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32) if scaled_boxes else torch.zeros((0, 4))
        
     
        
        return {
        'image': image_tensor,
        'boxes': boxes_tensor,
        'original_quads': original_quads,
        'texts': texts,
        'image_id': base_name,
        'scale_factors': torch.tensor(scale_factors),  # можешь добавить если нужно
        }   
    def _simple_ctpn_transform(self, image, bboxes=None):
        """
        ПРОСТОЙ CTPN ресайз - одна функция, всё делает.
        
        Args:
            image: RGB изображение (H, W, 3)
            bboxes: Список [[x1, y1, x2, y2], ...] или None
        
        Returns:
            Если bboxes=None: (resized_image, scale_factors)
            Если bboxes заданы: (resized_image, scaled_bboxes, scale_factors)
        """
        h, w = image.shape[:2]
        
        # 1. Вычисляем масштаб как в CTPN
        min_dim = min(h, w)
        max_dim = max(h, w)
        
        scale = 600.0 / min_dim  # минимальная сторона = 600px
        
        if np.round(scale * max_dim) > 1200:  # если максимальная > 1200px
            scale = 1200.0 / max_dim
        
        # 2. Новые размеры
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # 3. Делаем кратными 16 (для CTPN)
        new_h = ((new_h + 15) // 16) * 16
        new_w = ((new_w + 15) // 16) * 16
        
        # 4. Ресайзим изображение
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 5. Масштабные коэффициенты
        scale_h = new_h / h
        scale_w = new_w / w
        
        if bboxes is None:
            return resized_img, (scale_w, scale_h)
        
        # 6. Масштабируем bboxes если есть
        scaled_bboxes = []
        for bbox in bboxes:
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                scaled_bboxes.append([
                    x1 * scale_w,
                    y1 * scale_h,
                    x2 * scale_w,
                    y2 * scale_h
                ])
        
        return resized_img, scaled_bboxes, (scale_w, scale_h)
    




IMAGE_MEAN = [123.68, 116.779, 103.939]


class SROIEDataset2(Dataset):
    def __init__(self, split, image_dir, annotation_dir):
        """
        Args:
            split: 'train' или 'val'
            image_dir: путь к изображениям
            annotation_dir: путь к аннотациям
            target_size: оригинальный целевой размер (width, height)
            transform: Albumentations трансформации (если None, используется только ресайз)
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        
        # Определяем файл со списком изображений для split
        root = None
        if split == 'train':
            root = "../../data/splits/train_files.txt"
        elif split == 'val':
            root = "../../data/splits/val_files.txt"
        
        if root is not None and os.path.exists(root):
            with open(root, 'r', encoding='utf-8') as f:
                mask_files = [line.strip() for line in f if line.strip()]
            self.image_files = sorted([
                f for f in os.listdir(image_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f in mask_files
            ])
        else:
            self.image_files = sorted([
                f for f in os.listdir(image_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        h, w, c = image.shape

 
        
        # Загрузка аннотаций
        base_name = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.annotation_dir, f"{base_name}.txt")
        boxes = []  # Конвертированные в горизонтальные прямоугольники [x_min, y_min, x_max, y_max]

        
        # Читаем аннотации
        if os.path.exists(ann_path):
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(ann_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    content = lines
                    break
                except UnicodeDecodeError:
                    continue
            
            if content:
                for line_no, line in enumerate(content):
                    parts = line.strip().split(',', 8)
                    if len(parts) == 9:
                        # Оригинальный квадроугольник [x1, y1, x2, y2, x3, y3, x4, y4]
                        quad = list(map(float, parts[:8]))
                        text = parts[8]
                        
                        # Конвертируем в горизонтальный прямоугольник
                        x_coords = [quad[0], quad[2], quad[4], quad[6]]
                        y_coords = [quad[1], quad[3], quad[5], quad[7]]
                        rect = [
                            min(x_coords),  # x_min
                            min(y_coords),  # y_min
                            max(x_coords),  # x_max
                            max(y_coords)   # y_max
                        ]
                        
                        boxes.append(rect)
            gtbox = np.array(boxes)

            if np.random.randint(2) == 1:
                image = image[:, ::-1, :]
                newx1 = w - gtbox[:, 2] - 1
                newx2 = w - gtbox[:, 0] - 1
                gtbox[:, 0] = newx1
                gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)

        m_img = image - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr
     