import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def forward(self, sentences, tags, lengths):
        # sentences: (B, L)
        embeds = self.word_embeds(sentences)  # (B, L, E)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=True
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        emissions = self.hidden2tag(lstm_out)  # (B, L, num_tags)

        # CRF loss
        loss = -self.crf(emissions, tags, mask=sentences != 0, reduction='mean')
        return loss

    def predict(self, sentences, lengths):
        embeds = self.word_embeds(sentences)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=True
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        emissions = self.hidden2tag(lstm_out)
        predicts = self.crf.decode(emissions, mask=sentences != 0)
        return predicts