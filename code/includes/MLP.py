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
   
   def d_sigmoid(self,sig_x):
     return sig_x * (1.0 - sig_x)

   def d_hidden_layer(self):
     return self.d_sigmoid(self.hiddenNodes) * self.hiddenWeights


   def d_network(self):
     return self.d_sigmoid(self.outputNodes) * self.outputWeights * self.d_hidden_layer
     

   def dd_network(self):
     return self.d_sigmoid(self.d_network()) * self.outputWeights * self.d_sigmoid(self.d_hidden_layer()) * self.hiddenWeights ##//TO CHECK

   
   

   def process(self, inputArray):
     self.hiddenNodes = self.sigmoid(inputArray.dot(self.hiddenWeights) + self.hiddenBias)
     self.outputNodes = self.sigmoid(self.hiddenNodes.dot(self.outputWeights) + self.outputBias)

     return self.outputNodes
   
   def train(self, inputArray, targetOut, learningRate):
     error = targetOut - self.process(inputArray)
     loss = 0.5 * error * error
     

     delta = (error) * self.d_sigmoid(self.outputNodes) 
     updateBiasOut = delta
     
     updateWeightsOut = np.transpose(np.outer(delta , self.hiddenNodes))
     delta = delta.dot(np.transpose(self.outputWeights)) * (self.d_sigmoid(self.hiddenNodes))
     
     
     updateBiasHidden = delta
     updateWeightsHidden = np.transpose(np.outer(delta, inputArray))

     self.outputWeights += learningRate * updateWeightsOut
     self.outputBias += learningRate * updateBiasOut
     self.hiddenBias += learningRate * updateBiasHidden
     self.hiddenWeights += learningRate * updateWeightsHidden
     return loss
     
     
     
     
     
