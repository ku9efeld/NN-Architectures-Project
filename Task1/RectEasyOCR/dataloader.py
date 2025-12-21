import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

warnings.filterwarnings('ignore')


class ReceiptDataset(Dataset):
    def __init__(self, data_dir, split='train', image_size=(640, 640)):
        """
        Args:
            data_dir: Path to data directory (contains train/test folders)
            split: 'train' or 'test'
            image_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.image_size = image_size

        # Get all image files
        self.img_dir = self.data_dir / 'img'
        self.box_dir = self.data_dir / 'box'
        self.entries_dir = self.data_dir / 'entries'

        self.image_files = sorted(list(self.img_dir.glob('*.jpg')) +
                                  list(self.img_dir.glob('*.png')))

        print(f"Found {len(self.image_files)} images in {split} set")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image_id = img_path.stem

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # Resize image
        image_resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # Load bounding boxes
        box_path = self.box_dir / f"{image_id}.txt"
        boxes, texts = self.load_boxes(box_path, original_w, original_h)

        # Convert boxes to detection format (score map)
        score_map = self.create_score_map(boxes)

        # Ensure all tensors are properly formatted
        return {
            'image': image_tensor,
            'image_id': image_id,
            'boxes': torch.tensor(boxes).float() if len(boxes) > 0 else torch.zeros((0, 8)),
            'texts': texts,
            'score_map': torch.from_numpy(score_map).float().unsqueeze(0),  # Add channel dimension
            'original_size': torch.tensor([original_h, original_w])
        }

    def load_boxes(self, box_path, original_w, original_h):
        """Load boxes from text file"""
        boxes = []
        texts = []

        if not box_path.exists():
            return boxes, texts

        try:
            with open(box_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            return boxes, texts

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 9:  # 8 coordinates + text
                try:
                    # Parse coordinates
                    coords = list(map(int, parts[:8]))
                    text = ','.join(parts[8:])  # Handle text with commas

                    # Convert to [x1, y1, x2, y2, x3, y3, x4, y4]
                    box = np.array(coords, dtype=np.float32)

                    # Normalize to [0, 1] range
                    box[::2] /= original_w  # x coordinates
                    box[1::2] /= original_h  # y coordinates

                    boxes.append(box)
                    texts.append(text)
                except:
                    continue

        return boxes, texts

    def create_score_map(self, boxes, sigma=5):
        """Create score map for text detection"""
        h, w = self.image_size
        score_map = np.zeros((h, w), dtype=np.float32)

        if len(boxes) == 0:
            return score_map

        for box in boxes:
            # Convert normalized coords to pixel coordinates
            pixel_box = box.copy()
            pixel_box[::2] *= w
            pixel_box[1::2] *= h

            # Create polygon mask
            polygon = pixel_box.reshape(4, 2).astype(np.int32)

            # Fill polygon
            cv2.fillPoly(score_map, [polygon], 1.0)

        # Apply Gaussian blur for soft boundaries
        score_map = cv2.GaussianBlur(score_map, (0, 0), sigma)
        score_map = np.clip(score_map, 0, 1)

        return score_map

    def visualize_sample(self, idx):
        """Visualize a sample with bounding boxes"""
        sample = self[idx]
        image = sample['image'].permute(1, 2, 0).numpy()
        boxes = sample['boxes'].numpy()
        texts = sample['texts']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot image with boxes
        ax1.imshow(image)
        if len(boxes) > 0:
            for i, (box, text) in enumerate(zip(boxes, texts)):
                # Convert normalized to pixel coordinates
                pixel_box = box.copy()
                pixel_box[::2] *= self.image_size[1]
                pixel_box[1::2] *= self.image_size[0]

                # Draw polygon
                polygon = pixel_box.reshape(4, 2)
                ax1.plot([polygon[0, 0], polygon[1, 0], polygon[2, 0], polygon[3, 0], polygon[0, 0]],
                         [polygon[0, 1], polygon[1, 1], polygon[2, 1], polygon[3, 1], polygon[0, 1]],
                         'r-', linewidth=2)

        ax1.set_title(f'Image with {len(boxes)} bounding boxes')
        ax1.axis('off')

        # Plot score map
        ax2.imshow(sample['score_map'][0].numpy(), cmap='hot')
        ax2.set_title('Score Map')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()