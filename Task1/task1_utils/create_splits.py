import os
import random

# Пути (относительно корня проекта)
TRAIN_IMG_DIR = "./data/train/img"
SPLIT_DIR = "./data/splits"

# Создаём папку для сплитов, если её нет
os.makedirs(SPLIT_DIR, exist_ok=True)

# Фиксируем seed для воспроизводимости
random.seed(42)

# Получаем список изображений
if not os.path.exists(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"Директория с изображениями не найдена: {TRAIN_IMG_DIR}")

images = [
    f for f in os.listdir(TRAIN_IMG_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
images = sorted(images)  # сортируем для детерминизма до shuffle
random.shuffle(images)

# Делим 90/10
n = len(images)
split_idx = int(0.9 * n)
train_files = images[:split_idx]
val_files = images[split_idx:]

# Сохраняем
with open(os.path.join(SPLIT_DIR, "train_files.txt"), "w") as f:
    f.write("\n".join(train_files))

with open(os.path.join(SPLIT_DIR, "val_files.txt"), "w") as f:
    f.write("\n".join(val_files))

print(f"  Общее количество изображений: {n}")
print(f"  Train: {len(train_files)}")
print(f"  Val:   {len(val_files)}")
print(f"  Списки сохранены в папку: {SPLIT_DIR}/")