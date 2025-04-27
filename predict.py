# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 03:39:23 2025

@author: ChitvanSingh
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
#import sys
#sys.path.append(r'E:\Academics_IISER\Sem8\IV_Process\notebook1')

#from config import hyper_param, input_shape
# replace MyCustomModel with the name of your model

#from model import CNNModel as TheModel

def predictor_airtemp_anomaly(model, weights_path, data_path, hyper_param, loss_fn=None):
    """
    Evaluates a CNN model using a test dataset loaded from a .npz file.
    
    Args:
        model (torch.nn.Module): Trained CNN model.
        weights_path: path to final weights of the trained CNN model.
        data_path (str): Path to the .npz file containing 'x_test' and 'y_test'.
        loss_fn (callable, optional): Loss function (default: nn.MSELoss).
        
    Returns:
        y_pred = predicted value
        y_true = Target value
        mae = mean absolute error
        mse = mean squared error
        pcc = pearson_correlation_coefficient
    """
    
    # Load the test data from .npz
    data = np.load(data_path)
    
    # extracting 
    x_test = data['input_img']  # (N, H, W, C)
    y_test = data['output_img']  # (N,)
    
    # Check if x_test is a single sample or a batch
    if len(x_test.shape) == 3:  # Single sample (H, W, C)
        x_test = np.expand_dims(x_test, axis=0)  # (1, H, W, C)
    
    N, H, W, C = x_test.shape
    # If y_test is a scalar (shape ()), convert it to an array (1,) for consistency
    if y_test.shape == ():  # Scalar (single value)
        y_test = np.expand_dims(y_test, axis=0)  # Convert to (1,)

    # Convert to tensors
    x_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    y_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)        # (N, 1)
    
    print(x_tensor.shape, y_tensor.shape)
    
    model = model(C, H, W, hyper_param[0])
    model.load_state_dict(torch.load(weights_path))
    # Activate evaluation mode
    model.eval()
    # Use default loss function if not provided
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    
    test_loss, test_mae, test_samples = 0.0, 0.0, 0
    all_preds, all_targets = [], []

    # loop over all input images (numpy arrays)
    with torch.no_grad():
        for i in range(x_tensor.size(0)):
            input_img = x_tensor[i].unsqueeze(0)   # shape: (1, C, H, W)
            target = y_tensor[i].unsqueeze(0)      # shape: (1, 1)

            output = model(input_img)
            loss = loss_fn(output, target)
            mae = torch.mean(torch.abs(output - target))

            test_loss += loss.item()
            test_mae += mae.item()
            test_samples += 1

            all_preds.append(output.squeeze().cpu())
            all_targets.append(target.squeeze().cpu())

    # Convert to numpy arrays
    y_pred = torch.stack(all_preds).numpy()
    y_true = torch.stack(all_targets).numpy()

    # Compute metrics
    results = {
        'mse': test_loss / test_samples,
        'mae': test_mae / test_samples,
        'predictions': y_pred,
        'true_values': y_true,
        'pcc': None
    }
    print(results)

    # Compute Pearson correlation coefficient if at least two samples are given
    if len(y_pred) >= 2:
        pcc_value, _ = pearsonr(y_true, y_pred)
        results['pcc'] = pcc_value

    return y_pred, y_true, results['mse'], results['mae'], results['pcc']


#predictor_airtemp_anomaly(TheModel, weights_path, data_path, hyper_param)

