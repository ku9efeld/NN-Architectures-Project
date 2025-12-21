# Task2/CRNN_CTC/utils/prepare_crops.py

import cv2
import pandas as pd
from pathlib import Path
import argparse
import shutil

def parse_box_line(line):
    """
    Парсит строку вида: 'x1,y1,x2,y2,x3,y3,x4,y4,text'
    Возвращает: ((x_min, y_min, x_max, y_max), text)
    """
    parts = line.strip().rsplit(',', maxsplit=1)
    if len(parts) != 2:
        return None, None
    coords_str, text = parts
    try:
        coords = list(map(int, coords_str.split(',')))
        if len(coords) != 8:
            return None, None
        x_coords = coords[0::2]
        y_coords = coords[1::2]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return (x_min, y_min, x_max, y_max), text
    except Exception:
        return None, None

def load_split_file(split_path):
    with open(split_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def prepare_crops(dataset_root, output_dir):
    """
    dataset_root: путь к папке с train/, test/, splits/ (обычно: ../../data)
    output_dir: куда сохранять кропы и CSV (обычно: ../../data)
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"

    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем сплиты
    splits_dir = dataset_root / "splits"
    train_names = load_split_file(splits_dir / "train_files.txt")
    val_names = load_split_file(splits_dir / "val_files.txt")

    # Авто-список test из test/img/
    test_img_dir = dataset_root / "test" / "img"
    test_names = set()
    if test_img_dir.exists():
        test_names = {p.name for p in test_img_dir.glob("*.jpg")}
        # Сохраняем test_files.txt для воспроизводимости
        test_split_path = splits_dir / "test_files.txt"
        if not test_split_path.exists():
            with open(test_split_path, 'w', encoding='utf-8') as f:
                for name in sorted(test_names):
                    f.write(name + '\n')

    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    records_train, records_val, records_test = [], [], []
    crop_id = 0

    # Обработка train
    for split_name, box_dir, img_dir, name_set in [
        ("train", dataset_root / "train" / "box", dataset_root / "train" / "img", train_names | val_names),
        ("test", dataset_root / "test" / "box", dataset_root / "test" / "img", test_names),
    ]:
        print(f"Обработка: {split_name}")
        for txt_path in box_dir.glob("*.txt"):
            img_name = f"{txt_path.stem}.jpg"
            if img_name not in name_set:
                continue

            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            with open(txt_path, 'r', encoding='latin1') as f:
                lines = f.readlines()

            for line in lines:
                box, text = parse_box_line(line)
                if box is None or not text.strip():
                    continue

                x_min, y_min, x_max, y_max = box
                h, w = image.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                if x_max <= x_min or y_max <= y_min:
                    continue
                if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                    continue

                crop = image[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    continue

                crop_name = f"crop_{crop_id:06d}.png"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                rel_path = crop_path.relative_to(output_dir)
                record = {"filename": str(rel_path), "text": text}

                if split_name == "train":
                    if img_name in train_names:
                        records_train.append(record)
                    elif img_name in val_names:
                        records_val.append(record)
                else:  # test
                    records_test.append(record)

                crop_id += 1

    # Сохраняем CSV
    pd.DataFrame(records_train).to_csv(output_dir / "train.csv", index=False, encoding='utf-8')
    pd.DataFrame(records_val).to_csv(output_dir / "val.csv", index=False, encoding='utf-8')
    pd.DataFrame(records_test).to_csv(output_dir / "test.csv", index=False, encoding='utf-8')

    print(f"\n✅ Готово:")
    print(f"  Train: {len(records_train)} → train.csv")
    print(f"  Val:   {len(records_val)} → val.csv")
    print(f"  Test:  {len(records_test)} → test.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка кропов для OCR из data/train/ и data/test/")
    parser.add_argument("--dataset_root", type=str, default="../../data", help="Путь к data/ (по умолчанию: ../../data)")
    args = parser.parse_args()

    prepare_crops(args.dataset_root, args.dataset_root)