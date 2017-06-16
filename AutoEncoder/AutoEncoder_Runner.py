# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:55:52 2017

@author: Harry Shi
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from AutoEncoder import Autoencoder

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#batch_xs, batch_ys = mnist.train.next_batch(128)

autoencoder = Autoencoder(num_input = 784,
                          num_hidden_1 = 256,
                          num_hidden_2 = 128,
                          act_func = tf.nn.sigmoid,
                          optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.01))

training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(training_epochs):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        loss = autoencoder.model_training(batch_xs)
        loss_1 = autoencoder.get_loss(batch_xs)
    if epoch % display_step == 0:
        print("Epoch:",'%04d' %(epoch+1),"cost=","{:.9f}".format(loss),"cost2=","{:.9f}".format(loss_1))
            
print("Opt Done!")

image_reconstruction = autoencoder.reconstruct(mnist.train.images[:examples_to_show])
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.train.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(image_reconstruction[i], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()