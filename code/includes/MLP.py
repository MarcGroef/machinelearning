import numpy as np


class MLP():
  
   nLayers = 2
   
   def __init__(self, nInputNodes, hiddenSize, outputSize):
     self.hiddenWeights = np.random.rand(nInputNodes, hiddenSize) * 0.0001

     self.outputWeights = np.random.rand(hiddenSize, outputSize) * 0.0001
     self.hiddenBias = np.ones(hiddenSize) * -1
     self.outputBias = np.ones(outputSize) * -1
     pass

   def activation(self, x):
     #return self.sigmoid(x)
     return self.tanh(x)

   def d_activation(self, act_x):
     #return self.d_sigmoid(act_x)
     return self.d_tanh(act_x)
   
   def sigmoid(self,x):
     return 1.0 / (1.0 + np.exp(-1. * x))
   
   def d_sigmoid(self,sig_x):
     return sig_x * (1.0 - sig_x)

   def tanh(self, x):
     return np.tanh(x)

   def d_tanh(self, tanh_x):
     return 1 - tanh_x * tanh_x

   def d_hidden_layer(self):
     return self.d_sigmoid(self.hiddenNodes) * self.hiddenWeights


   def d_network(self):
     #print self.d_sigmoid(self.outputNodes).dot(np.transpose(self.outputWeights))
     return self.d_sigmoid(self.outputNodes).dot(np.transpose(self.outputWeights)) * self.d_hidden_layer
     

   def dd_network(self):
     return self.d_sigmoid(self.d_network()) * self.outputWeights * self.d_sigmoid(self.d_hidden_layer()) * self.hiddenWeights ##//TO CHECK

   def process(self, inputArray):
     self.hiddenNodes = self.activation(inputArray.dot(self.hiddenWeights) + self.hiddenBias)
     self.outputNodes = self.activation(self.hiddenNodes.dot(self.outputWeights) + self.outputBias)

     return self.outputNodes
   
   def train(self, inputArray, targetOut, learningRate):
     error = targetOut - self.process(inputArray)
     loss = 0.5 * error * error
     

     delta = (error) * self.d_activation(self.outputNodes) 
     updateBiasOut = delta
     
     updateWeightsOut = np.transpose(np.outer(delta , self.hiddenNodes))
     delta = delta.dot(np.transpose(self.outputWeights)) * (self.d_activation(self.hiddenNodes))
     
     
     updateBiasHidden = delta
     updateWeightsHidden = np.transpose(np.outer(delta, inputArray))

     self.outputWeights += learningRate * updateWeightsOut
     self.outputBias += learningRate * updateBiasOut
     self.hiddenBias += learningRate * updateBiasHidden
     self.hiddenWeights += learningRate * updateWeightsHidden
     return loss
     
     
     
     
     
