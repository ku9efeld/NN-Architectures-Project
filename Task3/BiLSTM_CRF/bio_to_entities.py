import json
from pathlib import Path

def bio_to_entities(tokens, tags):
    """
    Преобразует токены и BIO-теги в словарь сущностей.
    """
    entities = {
        "company": "",
        "date": "",
        "address": "",
        "total": ""
    }
    
    current_phrase = []
    current_label = None

    for token, tag in zip(tokens, tags):
        if tag == "O":
            # Завершаем текущую сущность
            if current_label and current_label in entities:
                entities[current_label] = " ".join(current_phrase)
            current_phrase = []
            current_label = None
        elif tag.startswith("B-"):
            # Завершаем предыдущую сущность
            if current_label and current_label in entities:
                entities[current_label] = " ".join(current_phrase)
            # Начинаем новую
            current_label = tag[2:]  # "company", "date", etc.
            current_phrase = [token]
        elif tag.startswith("I-"):
            label = tag[2:]
            if current_label == label:
                current_phrase.append(token)
            else:
                # Некорректный тег — завершаем текущую
                if current_label and current_label in entities:
                    entities[current_label] = " ".join(current_phrase)
                current_phrase = []
                current_label = None
        else:
            # Неизвестный тег — игнорируем
            if current_label and current_label in entities:
                entities[current_label] = " ".join(current_phrase)
            current_phrase = []
            current_label = None

    # Не забыть последнюю сущность
    if current_label and current_label in entities:
        entities[current_label] = " ".join(current_phrase)

    return entities

def convert_predictions_to_submission(predictions_json, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(predictions_json, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    for item in predictions:
        sample_id = item["id"]
        tokens = item["tokens"]
        tags = item["ner_tags"]

        entities = bio_to_entities(tokens, tags)

        output_file = output_dir / f"{sample_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for key in ["company", "date", "address", "total"]:
                f_out.write(f"{key}: {entities[key]}\n")

    print(f"Конвертация завершена! {len(predictions)} файлов сохранено в {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Путь к predictions.json")
    parser.add_argument("output_dir", help="Папка для сохранения .txt файлов")
    args = parser.parse_args()

    convert_predictions_to_submission(args.input_json, args.output_dir)