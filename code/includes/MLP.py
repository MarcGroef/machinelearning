import numpy as np


class MLP():
  
   nLayers = 2
   
   
   
   def __init__(self, nInputNodes, hiddenSize, outputSize):
     self.hiddenWeights = np.random.rand(nInputNodes, hiddenSize) 
     self.outputWeights = np.random.rand(hiddenSize, outputSize) 
     self.hiddenBias = np.ones(hiddenSize) * -1
     self.outputBias = np.ones(outputSize) * -1
     pass
   
   def sigmoid(self,x):
     return 1.0 / (1.0 + np.exp(-1. * x))
   
   def d_sigmoid(self,x):
     sig = self.sigmoid(x)
     return sig * (1. - sig)
   
   def relu(self, x):
     return np.maximum(0, x)
   
   def d_relu(self, x):
     #print "d_relu" + str(x)
     #print np.where(x > 0, 1, 0)
     return  np.where(x > 0, 1, 0)
   
   def tanh(self, x):
     return np.tanh(x)
   
   def d_tanh(self, x):
     return np.tanh(x) * np.tanh(x)
   
   def process(self, inputArray):
     self.hiddenSum = inputArray.dot(self.hiddenWeights) + self.hiddenBias
     self.hiddenNodes = self.sigmoid(self.hiddenSum)
     
     self.outputSum = self.hiddenNodes.dot(self.outputWeights) + self.outputBias
     self.outputNodes = self.sigmoid(self.outputSum)
     return self.outputNodes
   
   def train(self, inputArray, targetOut, learningRate):
     error = targetOut - self.process(inputArray)
     loss = 0.5 * error * error
     

     delta = (error) * self.d_sigmoid(self.outputSum) 
     updateBiasOut = delta
     
     updateWeightsOut = np.transpose(np.outer(delta , self.hiddenNodes))
     delta = delta.dot(np.transpose(self.outputWeights)) * (self.d_sigmoid(self.hiddenSum))
     
     
     updateBiasHidden = delta
     updateWeightsHidden = np.transpose(np.outer(delta, inputArray))

     self.outputWeights += learningRate * updateWeightsOut
     self.outputBias += learningRate * updateBiasOut
     self.hiddenBias += learningRate * updateBiasHidden
     self.hiddenWeights += learningRate * updateWeightsHidden
     return loss
     
     
     
     
     