import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class Task3Dataset(Dataset):
    def __init__(self, json_path, word2idx=None, tag2idx=None, min_freq=1):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Собираем частоты слов
        word_freq = {}
        for item in self.data:
            for word in item["tokens"]:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Создаём словари
        if word2idx is None:
            self.word2idx = {"<PAD>": 0, "<UNK>": 1}
            idx = 2
            for word, freq in word_freq.items():
                if freq >= min_freq:
                    self.word2idx[word] = idx
                    idx += 1
        else:
            self.word2idx = word2idx

        if tag2idx is None:
            all_tags = set()
            for item in self.data:
                for tag in item["ner_tags"]:
                    all_tags.add(tag)
            self.tag2idx = {tag: i for i, tag in enumerate(sorted(all_tags))}
        else:
            self.tag2idx = tag2idx

        self.idx2tag = {i: tag for tag, i in self.tag2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["tokens"]
        tags = item["ner_tags"]

        word_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        tag_ids = [self.tag2idx[t] for t in tags]

        return {
            "words": torch.tensor(word_ids, dtype=torch.long),
            "tags": torch.tensor(tag_ids, dtype=torch.long),
            "length": len(word_ids)
        }

def collate_fn(batch):
    words = [item["words"] for item in batch]
    tags = [item["tags"] for item in batch]
    lengths = [item["length"] for item in batch]

    # Сортируем по длине (для pack_padded_sequence)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    words = [words[i] for i in sorted_indices]
    tags = [tags[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]

    words_padded = pad_sequence(words, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)

    return words_padded, tags_padded, torch.tensor(lengths)

def get_dataloaders(train_json, val_json, test_json, batch_size=16, min_freq=1):
    train_dataset = Task3Dataset(train_json, min_freq=min_freq)
    val_dataset = Task3Dataset(val_json, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx)
    test_dataset = Task3Dataset(test_json, word2idx=train_dataset.word2idx, tag2idx=train_dataset.tag2idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader,  train_dataset.word2idx, train_dataset.tag2idx, train_dataset.idx2tag