import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from string import ascii_uppercase, digits, punctuation

from Task1.task1_utils.ctpn_utils import cal_rpn

valid_chars = ascii_uppercase + digits + punctuation + " \t\n"


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

        # Параметры ресайза
        new_h, new_w = 640, 640  # задайте нужные размеры
        scale_x = new_w / w
        scale_y = new_h / h

        # Ресайз изображения
        image = cv2.resize(image, (new_w, new_h))

        # Загрузка аннотаций
        base_name = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.annotation_dir, f"{base_name}.txt")
        boxes = []

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
                        quad = list(map(float, parts[:8]))
                        
                        # Масштабирование координат квадроугольника
                        for i in range(0, 8, 2):
                            quad[i] *= scale_x
                            quad[i+1] *= scale_y
                        
                        x_coords = [quad[0], quad[2], quad[4], quad[6]]
                        y_coords = [quad[1], quad[3], quad[5], quad[7]]
                        rect = [
                            min(x_coords),  # x_min
                            min(y_coords),  # y_min
                            max(x_coords),  # x_max
                            max(y_coords)   # y_max
                        ]
                        
                        boxes.append(rect)

        gtbox = np.array(boxes) if boxes else np.array([])

        # if np.random.randint(2) == 1: # Отзеркаливание картинки
        #     image = image[:, ::-1, :]
        #     newx1 = new_w - gtbox[:, 2] - 1
        #     newx2 = new_w - gtbox[:, 0] - 1
        #     gtbox[:, 0] = newx1
        #     gtbox[:, 2] = newx2

        # Используем новые размеры для расчета RPN
        [cls, regr], _ = cal_rpn((new_h, new_w), (int(new_h / 8), int(new_w / 8)), 8, gtbox)

        m_img = image - IMAGE_MEAN
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr, gtbox