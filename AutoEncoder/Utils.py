# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:02 2017

@author: Harry Shi
"""

import numpy as np

def get_random_block_from_data(data, batch_size):
    n_samples, n_features = data.shape
    start_index = np.random.randint(0, n_samples - batch_size)
    return data[start_index:(start_index + batch_size)]