# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:06:11 2017

@author: Harry Shi

Run auto encoder with .mat form input
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from AutoEncoder import Autoencoder
from scipy import io
import Utils

dataname = r'.\data\MNIST_70000n_784d_10c_full'
data = io.loadmat(dataname+'.mat')
fea = data['fea']
n_samples, n_features = fea.shape
print("Number of Samples:", '%d' % n_samples, ", Number of Features:", '%d' % n_features)


autoencoder = Autoencoder(num_input = n_features,
                          num_hidden_1 = 256,
                          num_hidden_2 = 128,
                          act_func = tf.nn.relu,
                          optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.01))

training_epochs = 20
batch_size = 128
display_step = 1
examples_to_show = 10
total_batch = int(n_samples/batch_size)

for epoch in range(training_epochs):
    for i in range(total_batch):
        batch_xs = Utils.get_random_block_from_data(fea,batch_size)
        loss = autoencoder.model_training(batch_xs)
        loss_1 = autoencoder.get_loss(batch_xs)
    if epoch % display_step == 0:
        print("Epoch:",'%04d' %(epoch+1),"cost=","{:.9f}".format(loss),"cost2=","{:.9f}".format(loss_1))
            
print("Opt Done!")

#image_reconstruction = autoencoder.reconstruct(fea[:examples_to_show])
#f, a = plt.subplots(2, 10, figsize=(10, 2))
#for i in range(examples_to_show):
#    a[0][i].imshow(np.reshape(fea[i], (28, 28)))
#    a[1][i].imshow(np.reshape(image_reconstruction[i], (28, 28)))
#f.show()
#plt.draw()
#plt.waitforbuttonpress()

fea_new = autoencoder.get_representation(fea)
io.savemat(dataname+'_feanew.mat',{'fea':fea_new,'gnd':data['gnd']})