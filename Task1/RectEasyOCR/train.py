import torch.optim as optim
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


class TextDetectorTrainer:
    def __init__(self, model, device='mps'):
        self.model = model
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Training on device: {self.device}")

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        # Training history
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        batch_count = 0

        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            score_maps = batch['score_map'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred_scores = self.model(images)

            # Calculate loss
            loss = self.criterion(pred_scores, score_maps)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / batch_count:.4f}'
            })

        return total_loss / batch_count

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        batch_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                score_maps = batch['score_map'].to(self.device)

                pred_scores = self.model(images)
                loss = self.criterion(pred_scores, score_maps)

                total_loss += loss.item()
                batch_count += 1

        return total_loss / batch_count if batch_count > 0 else 0

    def train(self, train_loader, val_loader, epochs=20):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)

            # Update learning rate
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss == min(self.history['val_loss']):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, 'best_detector_model.pth')
                print(f"Saved best model with val_loss: {val_loss:.4f}")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, f'checkpoint_epoch_{epoch + 1}.pth')

    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(self.history['val_loss'], label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png', dpi=150)
        plt.show()