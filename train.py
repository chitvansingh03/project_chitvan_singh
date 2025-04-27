# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 13:58:42 2025

@author: ChitvanSingh
"""
import numpy as np
import matplotlib.pyplot as plt
#from config import hyper_param
import torch
from torch import nn
from torch.optim import Adam
from torchsummary import summary

def train_model(input_shape, model, hyper_param, train_loader, val_loader):
    # Unpack input shape and hyperparameters
    C, H, W = input_shape
    drop_prob = hyper_param[0]  # Assuming hyper_params contains dropout probability
    epochs = hyper_param[2]
    # Instantiate the model
    model = model(input_channels=C, input_height=H, input_width=W, drop_prob=drop_prob)

    # Print model summary (input size is (channels, height, width) — note: batch size excluded)
    summary(model, input_size=(C, H, W))

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=hyper_param[1])  # Assuming hyper_params[1] is learning rate

    print(f' optimiser is Adam with learning rate = {hyper_param[1]}, loss = MSE')
    # History for plotting
    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": []
    }

    print(f"Starting training for {epochs} epochs")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss, train_mae, train_samples = 0.0, 0.0, 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets))
            loss.backward()
            optimizer.step()

            bsz = inputs.size(0)
            train_loss += loss.item() * bsz
            train_mae += mae.item() * bsz
            train_samples += bsz

        # Validation loop
        model.eval()
        val_loss, val_mae, val_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mae = torch.mean(torch.abs(outputs - targets))

                bsz = inputs.size(0)
                val_loss += loss.item() * bsz
                val_mae += mae.item() * bsz
                val_samples += bsz

        # Epoch averages
        train_loss_avg = train_loss / train_samples
        train_mae_avg = train_mae / train_samples
        val_loss_avg = val_loss / val_samples
        val_mae_avg = val_mae / val_samples

        # Save to history
        history["train_loss"].append(train_loss_avg)
        history["train_mae"].append(train_mae_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_mae"].append(val_mae_avg)

        # Print the results for the epoch
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss_avg:.4f}, MAE: {train_mae_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f}, MAE: {val_mae_avg:.4f}")

    # Final training result
    print(f"Training is complete. Final training loss (last 10 epochs): "
          f"{sum(history['train_loss'][-10:]) / 10:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["val_loss"], label='Validation Loss')
    plt.title(f"MSE Loss over Epochs,dp = {hyper_param[0]}, lr = {hyper_param[1]}, epoch = {hyper_param[2]}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # MAE Plot
    plt.subplot(1, 2, 2)
    plt.plot(history["train_mae"], label='Train MAE')
    plt.plot(history["val_mae"], label='Validation MAE')
    plt.title(f"MAE over Epochs, vs = {hyper_param[3]}, bsz = {hyper_param[4]}")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.show()


