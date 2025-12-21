# Task2/CRNN_CTC/dataset.py

import os
from pathlib import Path
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

# ==============================================================================
# LabelConverter: строка ↔ индексы (с blank-символом для CTC)
# ==============================================================================

class LabelConverter:
    def __init__(self, alphabet=None):
        # Алфавит SROIE: цифры, заглавные буквы, пунктуация
        if alphabet is None:
            alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.,-/'()&$%:# +"
        self.alphabet = alphabet
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(alphabet)}  # blank = 0
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(alphabet)}
        self.num_classes = len(alphabet) + 1  # +1 для blank-символа

    def encode(self, text):
        """Преобразует строку в список индексов."""
        try:
            return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        except KeyError as e:
            raise KeyError(f"Символ {e} отсутствует в алфавите. Алфавит: {self.alphabet}")

    def decode(self, indices):
        """Greedy decoding: удаляет blank и повторы."""
        text = ""
        prev_idx = 0  # blank = 0
        for idx in indices:
            if idx != 0 and idx != prev_idx:
                text += self.idx_to_char.get(idx, "")
            prev_idx = idx
        return text

# ==============================================================================
# OCR Dataset для SROIE
# ==============================================================================

class SROIEOCRDataset(Dataset):
    def __init__(self, csv_path, img_root):
        """
        csv_path: путь к train.csv / val.csv / test.csv
        img_root: путь к папке data/ (где лежат crops/)
        """
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.label_converter = LabelConverter()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / row["filename"]
        text = row["text"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, text

    def get_label_converter(self):
        return self.label_converter

# ==============================================================================
# Collate function для CTC
# ==============================================================================

def ocr_collate_fn(batch):
    images, texts = zip(*batch)
    label_converter = LabelConverter()

    # --- Тексты → индексы ---
    targets = []
    target_lengths = []
    for text in texts:
        label = label_converter.encode(text)
        targets.extend(label)
        target_lengths.append(len(label))
    targets = torch.IntTensor(targets)
    target_lengths = torch.IntTensor(target_lengths)

    # --- Изображения → тензоры ---
    processed_imgs = []
    input_lengths = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        new_w = max(int(w * 32 / h), 1)
        resized = cv2.resize(gray, (new_w, 32), interpolation=cv2.INTER_CUBIC)
        img_tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0) / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5  # [1, 32, new_w]
        processed_imgs.append(img_tensor)

        seq_len = new_w // 4 - 1
        if seq_len < 1:
            seq_len = 1
        input_lengths.append(seq_len)

    max_width = max(img.shape[2] for img in processed_imgs)
    batch_size = len(processed_imgs)
    padded_images = torch.full(
        (batch_size, 1, 32, max_width),
        fill_value=0.0,
        dtype=torch.float32
    )
    for i, img in enumerate(processed_imgs):
        padded_images[i, :, :, :img.shape[2]] = img

    input_lengths = torch.IntTensor(input_lengths)
    return padded_images, targets, input_lengths, target_lengths