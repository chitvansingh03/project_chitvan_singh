# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:16:05 2025

@author: ChitvanSingh
"""

# dataset.py
import sys
sys.path.append(r'E:\Academics_IISER\Sem8\IV_Process\notebook1')

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from config import hyper_param # Importing the config file to get hyper_param

class UnicornImgDataset:
    def __init__(self, data_path, batch_size=32, val_split=0.1, seed=42):
        """
        Dataset for loading and splitting unicorn image data for training and validation.

        Args:
        - data_path (str): Path to the .npz file containing the dataset.
        - batch_size (int): The batch size for loading data.
        - val_split (float): Proportion of data to use for validation.
        - seed (int): Random seed for reproducibility.
        """
        # Filepath and hyperparameters
        self.npz_path = data_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed

        # Load data
        data = np.load(self.npz_path)
        
        # Ensure keys exist in the loaded data
        if 'x_train' not in data or 'y_train' not in data or 'var_name' not in data:
            raise ValueError("The npz file must contain 'x_train', 'y_train', and 'var_name' keys.")
        
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.var_names = data['var_name']

        # Convert to tensors and correct shapes
        self.x_train_tensor = torch.tensor(self.x_train, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1)
        
        # Get dimensions
        self.N, self.C, self.H, self.W = self.x_train_tensor.shape

        # Create dataset and split into training and validation sets
        dataset = TensorDataset(self.x_train_tensor, self.y_train_tensor)
        train_size = int((1 - self.val_split) * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(self.seed)

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        # Data loaders for training and validation
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        """
        Returns the data loaders for training and validation.

        Returns:
        - train_loader (DataLoader): Data loader for the training data.
        - val_loader (DataLoader): Data loader for the validation data.
        """
        return self.train_loader, self.val_loader

    def get_datasets(self):
        """
        Returns the training and validation datasets.

        Returns:
        - train_dataset (Dataset): Training dataset.
        - val_dataset (Dataset): Validation dataset.
        """
        return self.train_dataset, self.val_dataset

def unicornLoader(npz_path, seed=42):
    """
    Utility function to instantiate and return a UnicornImgDataset.

    Args:
    - npz_path (str): Path to the .npz file containing the dataset.
    - seed (int): Random seed for reproducibility.

    Returns:
    - train_loader (DataLoader): Data loader for the training data.
    - val_loader (DataLoader): Data loader for the validation data.
    - train_dataset (Dataset): The training dataset.
    - val_dataset (Dataset): The validation dataset.
    """
    # Get batch_size and val_split from config.hyper_param
    batch_size = hyper_param[4]  # hyper_param[4] is batch_size
    val_split = hyper_param[3]   # hyper_param[3] is val_split

    # Create the dataset object
    dataset = UnicornImgDataset(npz_path, batch_size=batch_size, val_split=val_split, seed=seed)
    return dataset.get_loaders(), dataset.get_datasets()

#unicornLoader("E:\\Academics_IISER\\Sem8\\IV_Process\\test_train_data\\train_test_data.npz")