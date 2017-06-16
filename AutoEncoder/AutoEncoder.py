# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:12:04 2017

@author: Harry Shi

A stacked auto-encoder
"""

import tensorflow as tf

class Autoencoder(object):
    def __init__(self, num_input,num_hidden_1,num_hidden_2,act_func, optimizer = tf.train.AdamOptimizer()):
        # num_input: the dimension of the input data
        # num_hidden_1: the number of hidden units of the first layer
        # num_hidden_2: the number of hidden units of the second layer
        # act_func: activation function
        # optimization algorithm
        self.num_input = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.act_func = act_func
        
        weights = {
        'e_h1': tf.Variable(tf.random_normal([num_input,num_hidden_1])),
        'e_h2': tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
        'd_h1': tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1])),
        'd_h2': tf.Variable(tf.random_normal([num_hidden_1,num_input])),
                                                        }
        biases = {
         'e_b1': tf.Variable(tf.random_normal([num_hidden_1])),
         'e_b2': tf.Variable(tf.random_normal([num_hidden_2])),
         'd_b1': tf.Variable(tf.random_normal([num_hidden_1])),
         'd_b2': tf.Variable(tf.random_normal([num_input])),
                  }
        self.net_weights = weights
        self.net_biases = biases
        
        # model
        self.x = tf.placeholder(tf.float32,[None,self.num_input])
        self.e_hidden_1 = self.act_func(tf.add(tf.matmul(self.x, self.net_weights['e_h1']),self.net_biases['e_b1']))
        self.e_hidden_2 = self.act_func(tf.add(tf.matmul(self.e_hidden_1,self.net_weights['e_h2']),self.net_biases['e_b2']))
        self.d_hidden_1 = self.act_func(tf.add(tf.matmul(self.e_hidden_2,self.net_weights['d_h1']),self.net_biases['d_b1']))
        self.reconstuction = self.act_func(tf.add(tf.matmul(self.d_hidden_1,self.net_weights['d_h2']),self.net_biases['d_b2']))
        
        #loss
        self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.reconstuction, self.x),2))
        
        #optimizer
        self.optimizer = optimizer.minimize(self.loss)
        
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
        
        
    # model learning using data in x
    def model_training(self, x):
        opt, loss = self.sess.run((self.optimizer, self.loss), feed_dict = {self.x: x})
        return loss
        
    # get the loss w.r.t. x
    def get_loss(self, x):
        loss = self.sess.run(self.loss, feed_dict = {self.x: x})
        return loss
        
    # get the hidden representation of x
    def get_representation(self, x):
        new_fea = self.sess.run(self.e_hidden_2, feed_dict = {self.x: x})
        return new_fea
        
    # generate a sample givan a hidden representation
    def generate_sample_from_hidden(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.num_hidden_2]))
        return self.sess.run(self.reconstuction, feed_dict={self.e_hidden_2: hidden})
        
    # get a reconstruction of an input image/data sample
    def reconstruct(self, x):
        return self.sess.run(self.reconstuction, feed_dict = {self.x: x})
        
    # return the network weights
    def getNetWeights(self):
        return self.net_weights, self.net_biases
        
        