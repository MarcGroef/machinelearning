import tensorflow as tf
import numpy as np

class tfMLP():
   def __init__(self, nInputNodes, hiddenSize, outputSize):
      x_ = tf.placeholder(tf.float32, shape[1,nInputNodes])
      y_ = tf.placeholder(tf.float32, shape[1,1])

      hidden_weights = tf.Variable(tf.random_uniform([nInputNodes,hiddenSize]))
      hidden_bias = tf.Variable(tf.zeros([hiddenSize]))
      hidden_out = tf.matmul(x_, hidden_weights) + hidden_bias
      hidden_out = tf.nn.relu(hidden_out)

      out_weights = tf.Variable(tf.random_uniform([hiddenSize, 1]))
      out_bias = tf.Variable(tf.zeros([1]))
      self.out = tf.matmul(hidden_out, out_weights) + out_bias
      #out = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      error = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.out, y_))))

      self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(error)
      
      sess = tf.InteractiveSession()
      tf.global_variables_initializer().run()


   def train(self, inp, targ):
      sess.run(self.train_step, feed_dict={x_ : inp, y_ : targ})

   def predict(self, inp)
      return sess.run(self.out, feed_dict={x_ : inp)
