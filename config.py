# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 02:51:19 2025

@author: ChitvanSingh
"""

# Hyperparameters
drop_prob = 0.9
epochs = 100  # or your value
val_split = 0.1
batch_size = 32
learning_rate = 0.001
input_shape = [8, 15, 16] ## [Channels, rows, columns]

hyper_param = [drop_prob, learning_rate, epochs, val_split, batch_size]