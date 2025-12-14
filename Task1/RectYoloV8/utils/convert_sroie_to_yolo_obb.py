import os
import cv2
import shutil
from pathlib import Path

# ====== Пути ======
DATA_ROOT = "./data"
SPLIT = "train"
IMG_DIR = os.path.join(DATA_ROOT, SPLIT, "img")
BOX_DIR = os.path.join(DATA_ROOT, SPLIT, "box")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")

YOLO_OUT = "./Task1/RectYoloV8/yolo_obb_dataset"
IMG_OUT_TRAIN = os.path.join(YOLO_OUT, "images/train")
IMG_OUT_VAL = os.path.join(YOLO_OUT, "images/val")
LBL_OUT_TRAIN = os.path.join(YOLO_OUT, "labels/train")
LBL_OUT_VAL = os.path.join(YOLO_OUT, "labels/val")

os.makedirs(IMG_OUT_TRAIN, exist_ok=True)
os.makedirs(IMG_OUT_VAL, exist_ok=True)
os.makedirs(LBL_OUT_TRAIN, exist_ok=True)
os.makedirs(LBL_OUT_VAL, exist_ok=True)

# ====== Чтение сплитов ======
def read_split(filename):
    with open(os.path.join(SPLITS_DIR, filename)) as f:
        return [line.strip() for line in f if line.strip()]

train_files = [f for f in read_split("train_files.txt") if f]
val_files = [f for f in read_split("val_files.txt") if f]

# ====== Парсинг строки  ======
def parse_box_line(line: str):
    """Парсинг строки: x1,y1,...,x4,y4,text → [x1,y1,...,x4,y4], text"""
    parts = line.strip().split(',')
    if len(parts) < 8:
        return None, ""
    try:
        coords = list(map(float, parts[:8]))
        text = ','.join(parts[8:]) if len(parts) > 8 else ""
        return coords, text
    except ValueError:
        return None, ""

# ====== Конвертация ======
def convert_files(file_list, img_out, lbl_out):
    for img_name in file_list:
        base_name = Path(img_name).stem
        img_path = os.path.join(IMG_DIR, img_name)
        box_path = os.path.join(BOX_DIR, f"{base_name}.txt")

        if not os.path.exists(img_path) or not os.path.exists(box_path):
            continue

        # Копируем изображение
        shutil.copy2(img_path, os.path.join(img_out, img_name))

        # Читаем размер оригинального изображения
        img = cv2.imread(img_path)
        if img is None:
            print(f"Пропущено: {img_path} (не удалось прочитать)")
            continue
        h, w = img.shape[:2]

        # Читаем и парсим аннотации
        with open(box_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        label_lines = []
        for line in lines:
            coords, _ = parse_box_line(line)
            if coords and len(coords) == 8:
                # Нормализуем относительно оригинального размера
                norm_coords = [
                    coords[i] / (w if i % 2 == 0 else h)
                    for i in range(8)
                ]
                label_lines.append("0 " + " ".join(f"{c:.6f}" for c in norm_coords))

        # Сохраняем .txt с аннотациями
        lbl_name = Path(img_name).with_suffix(".txt")
        with open(os.path.join(lbl_out, lbl_name), "w") as f:
            f.write("\n".join(label_lines))

# ====== Запуск ======
print("Конвертация train...")
convert_files(train_files, IMG_OUT_TRAIN, LBL_OUT_TRAIN)

print("Конвертация val...")
convert_files(val_files, IMG_OUT_VAL, LBL_OUT_VAL)

print("Готово! Датасет в:", YOLO_OUT)