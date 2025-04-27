# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 02:51:36 2025

@author: ChitvanSingh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam

# Define the model with parameters
class CNNModel(nn.Module):
    def __init__(self, input_channels, input_height, input_width, drop_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(drop_prob)

        # Dynamically calculate the flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            dummy_out = self.pool(F.relu(self.conv1(dummy_input)))
            flattened_size = dummy_out.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer for regression

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
