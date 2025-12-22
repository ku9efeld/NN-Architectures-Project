import os
from pathlib import Path
import json
import argparse
import re
from difflib import SequenceMatcher
import json
from collections import Counter

def load_split_files(splits_dir):
    def read_txt(path):
        if not path.exists():
            return set()
        with open(path, 'r', encoding='utf-8') as f:
            return {Path(line.strip()).stem for line in f if line.strip()}
    train_files = read_txt(splits_dir / "train_files.txt")
    val_files = read_txt(splits_dir / "val_files.txt")
    return train_files, val_files

def parse_box_file(box_path):
    """
    Читает box-файл и возвращает список строк в порядке Y (сверху вниз).
    Каждый элемент — (y_min, text)
    """
    lines = []
    if not box_path.exists():
        return lines
    with open(box_path, 'r', encoding='latin1') as f:
        for line in f:
            parts = line.strip().rsplit(',', maxsplit=1)
            if len(parts) != 2:
                continue
            coords_str, text = parts
            try:
                coords = list(map(int, coords_str.split(',')))
                if len(coords) != 8:
                    continue
                y_coords = coords[1::2]
                y_min = min(y_coords)
                if text.strip():
                    lines.append((y_min, text.strip()))
            except Exception:
                continue
    # Сортируем по y_min (сверху вниз)
    lines.sort(key=lambda x: x[0])
    return [text for _, text in lines]

import json

def parse_entities_file(entities_path):
    if not entities_path.exists():
        return {}
    try:
        with open(entities_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Приводим ключи к нижнему регистру (опционально)
            return {k.lower(): v for k, v in data.items()}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Ошибка при парсинге {entities_path}: {e}")
        return {}


def create_bio_tags(lines, entities):
    tags = ["O"] * len(lines)

    company_gt = entities.get("company", "").strip()
    date_gt = entities.get("date", "").strip()
    total_gt = entities.get("total", "").strip()
    address_gt = entities.get("address", "").strip()

    # Вспомогательная функция: нормализация
    def normalize(s):
        return re.sub(r'[^a-zA-Z0-9]', ' ', s).lower().split()

    # === 1. COMPANY: выбираем строку с максимальным overlap по словам ===
    if company_gt:
        comp_words = normalize(company_gt)
        best_score = 0
        best_idx = -1
        for i, line in enumerate(lines):
            line_words = normalize(line)
            if not line_words:
                continue
            overlap = len(set(comp_words) & set(line_words))
            score = overlap / max(len(comp_words), 1)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_score >= 0.5:  # минимум 50% слов совпадает
            tags[best_idx] = "B-company"

    # === 2. DATE: ищем точное значение даты в любой строке ===
    if date_gt:
        for i, line in enumerate(lines):
            if date_gt in line:
                tags[i] = "B-date"
                break

    # === 3. TOTAL: ищем точное значение тотала в любой строке ===
    if total_gt:
        for i, line in enumerate(lines):
            if total_gt in line:
                tags[i] = "B-total"
                break

    # === 4. ADDRESS: ищем непрерывный блок строк, покрывающий адрес ===
    if address_gt:
        addr_words = normalize(address_gt)
        addr_word_set = set(addr_words)

        # Найдём все строки, которые содержат хоть одно уникальное слово из адреса
        candidate_indices = []
        word_freq = Counter(addr_words)
        rare_words = [w for w, cnt in word_freq.items() if cnt == 1 and len(w) > 2]

        for i, line in enumerate(lines):
            line_words = normalize(line)
            if not line_words:
                continue
            # Приоритет: редкие слова (уникальные для адреса)
            if any(w in line_words for w in rare_words):
                candidate_indices.append(i)
            elif len(set(line_words) & addr_word_set) >= 2:
                candidate_indices.append(i)

        # Выберем самый длинный непрерывный блок
        if candidate_indices:
            candidate_indices.sort()
            best_block = []
            current = [candidate_indices[0]]
            for idx in candidate_indices[1:]:
                if idx == current[-1] + 1:
                    current.append(idx)
                else:
                    if len(current) > len(best_block):
                        best_block = current
                    current = [idx]
            if len(current) > len(best_block):
                best_block = current

            if best_block:
                tags[best_block[0]] = "B-address"
                for j in best_block[1:]:
                    tags[j] = "I-address"

    return tags

def prepare_split(data_root, box_dir, entities_dir, file_list, output_path):
    dataset = []
    for img_name in sorted(list(file_list)):
        box_file = box_dir / f"{img_name}.txt"
        entities_file = entities_dir / f"{img_name}.txt"

        lines = parse_box_file(box_file)
        entities = parse_entities_file(entities_file)
        if not lines:
            continue

        tags = create_bio_tags(lines, entities)
        dataset.append({
            "id": img_name,
            "tokens": lines,
            "ner_tags": tags
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ {len(dataset)} примеров сохранено в {output_path}")

def prepare_test(data_root, output_path):
    data_root = Path(data_root)
    box_dir = data_root / "test" / "box"
    entities_dir = data_root / "test" / "entities"
    dataset = []
    for box_file in box_dir.glob("*.txt"):
        img_name = box_file.stem
        entities_file = entities_dir / f"{img_name}.txt"
        if not entities_file.exists():
            continue
        lines = parse_box_file(box_file)
        entities = parse_entities_file(entities_file)
        if not lines:
            continue
        tags = create_bio_tags(lines, entities)
        dataset.append({
            "id": img_name,
            "tokens": lines,
            "ner_tags": tags
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ {len(dataset)} тестовых примеров сохранено в {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для Task 3 (line-level, как у zzzDavid)")
    parser.add_argument("--data_root", type=str, default="../../data", help="Путь к data/")
    parser.add_argument("--output_dir", type=str, default=".", help="Папка для сохранения JSON")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    splits_dir = data_root / "splits"
    train_files, val_files = load_split_files(splits_dir)

    box_dir = data_root / "train" / "box"
    entities_dir = data_root / "train" / "entities"

    prepare_split(data_root, box_dir, entities_dir, train_files, output_dir / "train.json")
    prepare_split(data_root, box_dir, entities_dir, val_files, output_dir / "val.json")
    prepare_test(data_root, output_dir / "test.json")

if __name__ == "__main__":
    main()