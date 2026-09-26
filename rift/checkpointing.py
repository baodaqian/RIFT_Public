"""Checkpoint and loss-history persistence."""
import csv
import os
from datetime import datetime

import torch


def save_checkpoint(state, filepath):
    dirname = os.path.dirname(filepath)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        print(f"Created directory: {dirname}")
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    return checkpoint


def save_losses(training_loss, validation_loss, loss_path):
    with open(loss_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
        for epoch, (train_loss, val_loss) in enumerate(zip(training_loss, validation_loss), start=1):
            writer.writerow([epoch, train_loss, val_loss])


def generate_loss_path(base_dir, prefix="losses", extension=".csv"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}{extension}"
    return os.path.join(base_dir, filename)
