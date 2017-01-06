import tensorflow as tf
import numpy as np

class tfMLP():
   def __init__(self, nInputNodes, hiddenSize, outputSize):
      self.x_ = tf.placeholder(tf.float32, shape=[None, nInputNodes])
      self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

      hidden_weights = tf.Variable(tf.random_uniform([nInputNodes,hiddenSize]))
      hidden_bias = tf.Variable(tf.zeros([hiddenSize]))
      hidden_out = tf.matmul(self.x_, hidden_weights) + hidden_bias
      hidden_out = tf.nn.relu(hidden_out)

      out_weights = tf.Variable(tf.random_uniform([hiddenSize, 1]))
      out_bias = tf.Variable(tf.zeros([1]))
      self.out = tf.matmul(hidden_out, out_weights) + out_bias
      #out = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      error = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.out, self.y_))))

      self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(error)
      
      self.sess = tf.InteractiveSession()
      self.sess.run(tf.initialize_all_variables())


   def train(self, inp, targ):
      self.sess.run(self.train_step, feed_dict={self.x_ : inp, self.y_ : targ})

   def predict(self, inp):
      inp = [inp]
      return self.sess.run(self.out, feed_dict={self.x_ : inp})
