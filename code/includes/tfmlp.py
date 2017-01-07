import tensorflow as tf
import numpy as np

class tfMLP():
   def __init__(self, nInputNodes, hiddenSize, outputSize):
      self.x_ = tf.placeholder(tf.float32, shape=[None, nInputNodes])
      self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

      hidden_weights = tf.Variable(tf.truncated_normal([nInputNodes,hiddenSize], 0.001, 0.0001))
      hidden_bias = tf.Variable(tf.zeros([hiddenSize]))
      hidden_in = tf.matmul(self.x_, hidden_weights) + hidden_bias
      hidden_out = tf.tanh(hidden_in)

      out_weights = tf.Variable(tf.truncated_normal([hiddenSize,1], 0.05, 0.0001))
      out_bias = tf.Variable(tf.zeros([1]))
      self.out = tf.matmul(hidden_out, out_weights) + out_bias
      #out = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      self.error = 0.5 * tf.sqrt(tf.square(tf.sub(self.y_,self.out,)))

      self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.error)
      
      self.sess = tf.InteractiveSession()
      self.sess.run(tf.initialize_all_variables())


   def train(self, inp, targ):
      inp = np.reshape(inp, (-1, 2))
      targ = np.reshape(inp, (-1, 1))
      #print self.sess.run(self.error, feed_dict={self.x_ : inp, self.y_ : targ})
      self.sess.run(self.train_step, feed_dict={self.x_ : inp, self.y_ : targ})

   def process(self, inp):
      #inp 
      inp = np.reshape(inp, (-1, 2))
      return self.sess.run(self.out, feed_dict={self.x_ : inp})

   def close(self):
      self.sess.close()
