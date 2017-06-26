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
                
def get_block_from_data(data, batch_size, start_index):
    return data[start_index:(start_index + batch_size)]




def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in xrange(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w