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

IMG_DIR_TEST = os.path.join(DATA_ROOT, "test", "img")
BOX_DIR_TEST = os.path.join(DATA_ROOT, "test", "box")

YOLO_OUT = "./Task1/RectYoloV8/yolo_obb_dataset"
IMG_OUT_TRAIN = os.path.join(YOLO_OUT, "images/train")
IMG_OUT_VAL = os.path.join(YOLO_OUT, "images/val")
IMG_OUT_TEST = os.path.join(YOLO_OUT, "images/test")
LBL_OUT_TRAIN = os.path.join(YOLO_OUT, "labels/train")
LBL_OUT_VAL = os.path.join(YOLO_OUT, "labels/val")
LBL_OUT_TEST = os.path.join(YOLO_OUT, "labels/test")

for d in [IMG_OUT_TRAIN, IMG_OUT_VAL, IMG_OUT_TEST,
          LBL_OUT_TRAIN, LBL_OUT_VAL, LBL_OUT_TEST]:
    os.makedirs(d, exist_ok=True)

# ====== Чтение сплитов ======
def read_split(filename):
    with open(os.path.join(SPLITS_DIR, filename)) as f:
        return [line.strip() for line in f if line.strip()]

train_files = [f for f in read_split("train_files.txt") if f]
val_files = [f for f in read_split("val_files.txt") if f]

# ====== Парсинг строки  ======
def parse_box_line(line: str):
    parts = line.strip().split(',')
    if len(parts) < 8:
        return None
    try:
        return list(map(float, parts[:8]))
    except ValueError:
        return None

# ====== Конвертация ======
def convert_split(img_dir, box_dir, img_out, lbl_out, file_list=None):
    """
    Конвертирует набор файлов. Если file_list=None — берёт все изображения из img_dir.
    """
    if file_list is None:
        file_list = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    
    for img_name in file_list:
        base_name = Path(img_name).stem
        img_path = os.path.join(img_dir, img_name)
        box_path = os.path.join(box_dir, f"{base_name}.txt")

        if not os.path.exists(img_path) or not os.path.exists(box_path):
            continue

        shutil.copy2(img_path, os.path.join(img_out, img_name))

        img = cv2.imread(img_path)
        if img is None:
            print(f"Пропущено: {img_path}")
            continue
        h, w = img.shape[:2]

        with open(box_path, 'r', encoding='latin1') as f:
            lines = f.readlines()

        label_lines = []
        for line in lines:
            coords = parse_box_line(line)
            if coords and len(coords) == 8:
                norm_coords = [
                    coords[i] / (w if i % 2 == 0 else h)
                    for i in range(8)
                ]
                label_lines.append("0 " + " ".join(f"{c:.6f}" for c in norm_coords))

        lbl_name = Path(img_name).with_suffix(".txt")
        with open(os.path.join(lbl_out, lbl_name), "w") as f:
            f.write("\n".join(label_lines))

# ====== Запуск ======
print("Конвертация train...")
convert_split(IMG_DIR, BOX_DIR, IMG_OUT_TRAIN, LBL_OUT_TRAIN, train_files)

print("Конвертация val...")
convert_split(IMG_DIR, BOX_DIR, IMG_OUT_VAL, LBL_OUT_VAL, val_files)

print("Конвертация test...")
convert_split(IMG_DIR_TEST, BOX_DIR_TEST, IMG_OUT_TEST, LBL_OUT_TEST, file_list=None)

print(f"Датасет сохранён в: {YOLO_OUT}")