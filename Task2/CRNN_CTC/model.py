import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=32, hidden_size=256, num_lstm_layers=2):
        """
        CRNN модель для OCR с CTC loss.

        Аргументы:
            num_classes (int): размер алфавита + 1 (для blank)
            img_height (int): высота входного изображения (обычно 32)
            hidden_size (int): размер скрытого состояния LSTM
            num_lstm_layers (int): количество слоёв BiLSTM
        """
        super(CRNN, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # ==== CNN Backbone ====
        # (B, 1, H, W) → (B, 512, H//4, W//4) при H=32
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → H/2, W/2

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → H/4, W/4

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # → H/8, W/4

            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # → H/16, W/4

            # Layer 7
            nn.Conv2d(512, 512, kernel_size=2),  # (H/16 - 1) x (W/4 - 1)
            nn.ReLU(inplace=True),
        )

        # После CNN: H_out = (img_height // 16) - 1
        # Для img_height=32 → H_out = 1

        # ==== RNN ====
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        # ==== Head ====
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 из-за bidirectional


    def get_seq_len(self, input_length):
        """
        Вычисляет длину последовательности после CNN.
        input_length: ширина изображения (int)
        """
        # Layer 1: MaxPool2d(2,2) → /2
        seq_len = input_length // 2
        seq_len = seq_len // 2
        seq_len = seq_len - 1
        return seq_len
    
    def forward(self, x):
        """
        x: (B, 1, H, W) — грейскейл изображения
        Возвращает: (T, B, num_classes) — логиты для CTC
        """
        # CNN
        cnn_out = self.cnn(x)  # (B, 512, H', W')

        # Преобразуем в последовательность: (B, C, H, W) → (B, C*H, W)
        B, C, H, W = cnn_out.size()
        rnn_in = cnn_out.view(B, -1, W)  # (B, C*H, W)
        rnn_in = rnn_in.permute(0, 2, 1)  # (B, T=W, C*H)

        # RNN
        rnn_out, _ = self.rnn(rnn_in)  # (B, T, 2*hidden_size)

        # Head
        logits = self.fc(rnn_out)  # (B, T, num_classes)

        # CTC ожидает (T, B, num_classes)
        logits = logits.permute(1, 0, 2)  # (T, B, num_classes)

        return logits