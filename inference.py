# inference.py

import os
import cv2
import torch
import json
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms
import numpy as np

# --- Импорты моделей ---
# Model 1: YOLO
from Task1.RectYoloV8.model import get_yolov8_obb_model  
from Task1.task1_utils.metrics import polygon_to_bbox
# Model 2: CRNN
from Task2.CRNN_CTC.model import CRNN
from Task2.CRNN_CTC.dataset import LabelConverter
# Model 3: BiLSTM-CRF
from Task3.BiLSTM_CRF.model import BiLSTM_CRF
from Task3.BiLSTM_CRF.bio_to_entities import bio_to_entities  # или используем line-level логику ниже


# Цвета для разных сущностей (BGR)
ENTITY_COLORS = {
    "company": (0, 255, 255),   # жёлтый
    "date": (255, 0, 255),      # фиолетовый
    "total": (255, 255, 0),     # голубой
    "address": (0, 128, 255),   # оранжевый
    "O": (128, 128, 128)        # серый
}
# --- Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {DEVICE}")

# --- Загрузка моделей ---
def load_models():
    # 1. YOLO
    yolo = get_yolov8_obb_model(weights_path="Task1/RectYoloV8/best_model.pt")
    
    # 2. CRNN
    label_converter = LabelConverter()
    crnn = CRNN(num_classes=label_converter.num_classes, img_height=32, hidden_size=128, num_lstm_layers=1).to(DEVICE)
    crnn.load_state_dict(torch.load("Task2/CRNN_CTC/best_model.pt", map_location=DEVICE))
    crnn.eval()
    
    # 3. BiLSTM-CRF
    checkpoint = torch.load("Task3/BiLSTM_CRF/best_model.pth", map_location=DEVICE)
    config = checkpoint['config']
    bilstm = BiLSTM_CRF(
        vocab_size=len(checkpoint['word2idx']),
        tag_to_ix=checkpoint['tag2idx'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    ).to(DEVICE)

    bilstm.load_state_dict(checkpoint['model_state_dict'])
    bilstm.eval()

    word2idx = checkpoint['word2idx']
    idx2tag = checkpoint['idx2tag']
    
    return yolo, crnn, bilstm, label_converter, word2idx, idx2tag

# --- OCR строка за строкой ---
def recognize_line(image_crop, crnn_model, label_converter):
    if image_crop.size == 0:
        return ""
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 32))
    tensor = transforms.ToTensor()(gray).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = crnn_model(tensor)
        pred = label_converter.decode(logits.argmax(2).squeeze(1).cpu().numpy())
    return pred

# --- Line-level KIE ---
def extract_entities_line_level(lines, bilstm_model, word2idx, idx2tag, device=DEVICE):
    if not lines:
        return {"company": "", "date": "", "address": "", "total": ""}

    # Получаем предсказания
    word_ids = [word2idx.get(line, word2idx["<UNK>"]) for line in lines]
    words_tensor = torch.tensor(word_ids, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(lines)], dtype=torch.long)

    with torch.no_grad():
        pred_ids = bilstm_model.predict(words_tensor, lengths)[0]
        pred_tags = [idx2tag[idx] for idx in pred_ids]

    entities = bio_to_entities(lines, pred_tags)

    # Постобработка только для date и total
    import re
    if entities["date"]:
        match = re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', entities["date"])
        entities["date"] = match.group() if match else entities["date"]
    if entities["total"]:
        match = re.search(r'\d+\.\d{2}', entities["total"])
        entities["total"] = match.group() if match else entities["total"]

    return entities



def process_receipt(image_path, output_dir=None, save_visualization=True):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    vis_image = image.copy() if save_visualization else None
    lines = []
    obb_list = [] 

    # 1. Детекция
    yolo_results = yolo.predict(source=image_path)
    if not yolo_results or yolo_results[0] is None or yolo_results[0].obb is None:
        print("YOLO не обнаружил объектов")
        return {"company": "", "date": "", "address": "", "total": ""}

    obb = yolo_results[0].obb
    if len(obb) == 0:
        return {"company": "", "date": "", "address": "", "total": ""}

    # 2. OCR всех строк
    for obb_obj in obb:
        try:
            corners = obb_obj.xyxyxyxy[0].cpu().numpy()
            poly = corners.flatten()
            x1, y1, x2, y2 = polygon_to_bbox(poly)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            text = recognize_line(crop, crnn, label_converter)
            if text.strip():
                lines.append(text.strip())
                obb_list.append((obb_obj, text.strip(), y1))
        except Exception as e:
            print(f"Ошибка OCR: {e}")
            continue

    # Сортировка по Y
    obb_list.sort(key=lambda x: x[2])
    lines = [item[1] for item in obb_list]

    # 3. KIE — получаем теги
    if not lines:
        return {"company": "", "date": "", "address": "", "total": ""}

    word_ids = [word2idx.get(line, word2idx["<UNK>"]) for line in lines]
    words_tensor = torch.tensor(word_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(lines)], dtype=torch.long)

    with torch.no_grad():
        pred_ids = bilstm.predict(words_tensor, lengths)[0]
        pred_tags = [idx2tag[idx] for idx in pred_ids]

    entities = extract_entities_line_level(lines, bilstm, word2idx, idx2tag, DEVICE)

    # 4. Визуализация: рисуем боксы и подписываем сущности
    if save_visualization and vis_image is not None:
        for (obb_obj, text, _), tag in zip(obb_list, pred_tags):
            # Определяем метку сущности
            if tag == "O":
                label = "?"
                color = ENTITY_COLORS["O"]
            elif tag.startswith("B-") or tag.startswith("I-"):
                entity_type = tag.split("-")[1]
                label = entity_type.upper()  
                color = ENTITY_COLORS.get(entity_type, (0, 255, 0))
            else:
                label = "?"
                color = ENTITY_COLORS["O"]

            # Рисуем OBB
            corners = obb_obj.xyxyxyxy[0].cpu().numpy()
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], isClosed=True, color=color, thickness=2)

            # Позиция подписи — над боксом
            x_min = int(corners[:, 0].min())
            y_min = int(corners[:, 1].min())
            cv2.putText(
                vis_image,
                label,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Сохранение
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        vis_path = os.path.join(output_dir, f"vis_{os.path.basename(image_path)}")
        cv2.imwrite(vis_path, vis_image)
        print(f"Визуализация с сущностями сохранена: {vis_path}")

    return entities

# --- Запуск ---
"""if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Путь к изображению чека")
    parser.add_argument("--output", default="infer/result.json", help="Путь к выходному JSON")
    args = parser.parse_args()
    
    # Загружаем модели один раз
    yolo, crnn, bilstm, label_converter, word2idx, idx2tag = load_models()
    
    # Обработка
    result = process_receipt(args.image, "infer")
    
    # Сохранение
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Результат сохранён в {args.output}")
    print(json.dumps(result, ensure_ascii=False, indent=2)) """




def process_folder(input_dir, output_dir, save_visualization=True):
    """
    Обрабатывает все изображения в папке.
    
    Аргументы:
        input_dir (str): папка с изображениями (.jpg, .png)
        output_dir (str): папка для сохранения результатов
        save_visualization (bool): сохранять ли изображения с боксами
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Поддерживаемые форматы
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"В папке {input_dir} не найдено изображений")
        return

    print(f"Найдено {len(image_files)} изображений. Начинаем обработку...")

    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Обработка: {img_file.name}")
        
        try:
            # Обработка одного изображения
            entities = process_receipt(
                str(img_file),
                output_dir=str(output_path),
                save_visualization=save_visualization
            )
            
            sample_id = img_file.stem  # без расширения
            output_file = output_path / f"{sample_id}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for key in ["company", "date", "address", "total"]:
                    f.write(f"{key}: {entities[key]}\n")
                    
            print(f"Результат сохранён: {output_file.name}")
            
        except Exception as e:
            print(f"Ошибка при обработке {img_file.name}: {e}")

    print(f"\nГотово! Все результаты сохранены в {output_dir}")

# --- Запуск --- python inference.py data/test/img/ --output infer/ 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Путь к изображению или папке с изображениями")
    parser.add_argument("--output", default="infer/", help="Папка для результатов")
    parser.add_argument("--no-vis", action="store_true", help="Не сохранять визуализацию")
    args = parser.parse_args()

    # Загружаем модели один раз
    yolo, crnn, bilstm, label_converter, word2idx, idx2tag = load_models()

    input_path = Path(args.input)
    
    if input_path.is_file():
        # Один файл
        result = process_receipt(
            str(input_path),
            output_dir=args.output,
            save_visualization=not args.no_vis
        )
        sample_id = input_path.stem
        output_file = Path(args.output) / f"{sample_id}.txt"
        Path(args.output).mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for key in ["company", "date", "address", "total"]:
                f.write(f"{key}: {result[key]}\n")
        print(f"Результат сохранён: {output_file}")
        
    elif input_path.is_dir():
        # Папка
        process_folder(
            str(input_path),
            args.output,
            save_visualization=not args.no_vis
        )
    else:
        print(f"Путь не найден: {args.input}")