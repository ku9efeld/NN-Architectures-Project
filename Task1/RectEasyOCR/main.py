import train
import dataloader
import customEasyOCR

import torch
import numpy as np

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    BATCH_SIZE = 1  # Increased batch size for M4
    IMAGE_SIZE = (512, 512)  # Reduced size for faster training
    EPOCHS = 20

    # Disable pin_memory for MPS
    num_workers = 0  # Set to 0 for MPS compatibility

    # Create datasets
    print("Loading datasets...")
    train_dataset = dataloader.ReceiptDataset('/Users/kalex/projects/ITMO/Neural_Network_Architectures/project/data', split='train', image_size=IMAGE_SIZE)
    val_dataset = dataloader.ReceiptDataset('/Users/kalex/projects/ITMO/Neural_Network_Architectures/project/data', split='test', image_size=IMAGE_SIZE)

    # Test one sample to ensure it works
    print("Testing dataset...")
    sample = train_dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Score map shape: {sample['score_map'].shape}")
    print(f"Number of boxes: {len(sample['boxes'])}")

    # Create data loaders
    train_loader = dataloader.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,  # 0 for MPS
        pin_memory=False,  # False for MPS
        drop_last=True  # Avoid last incomplete batch
    )

    val_loader = dataloader.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    # Create model
    print("Creating model...")
    model = customEasyOCR.TextDetectionModel(pretrained=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = train.TextDetectorTrainer(model, device='mps')

    # Train
    trainer.train(train_loader, val_loader, epochs=EPOCHS)

    # Plot results
    trainer.plot_training_history()

    return trainer


if __name__ == '__main__':
    trainer = main()