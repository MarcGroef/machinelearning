import numpy as np


class MLP():
  
   nLayers = 2
   def __init__(self, nInputNodes, nLayers, hiddenSizes, outputSize):
     self.nLayers = nLayers
     self.hiddenSizes = hiddenSizes
     self.nInputNodes = nInputNodes
     self.hiddenSizes = hiddenSizes
     self.hiddenWeights = []
     self.hiddenBias = []
     self.hiddenNodes = []
     self.updateWeightsHidden = []
     self.updateBiasHidden = []
     self.hiddenInput = []

     for layer in range(nLayers):
         if layer == 0:
           self.hiddenWeights.append((np.random.rand(nInputNodes, hiddenSizes[layer]) - 0.5))
           self.updateWeightsHidden.append(np.zeros((nInputNodes, hiddenSizes[layer])))
         else:
           self.hiddenWeights.append((np.random.rand(hiddenSizes[layer - 1], hiddenSizes[layer]) - 0.5))
           self.updateWeightsHidden.append(np.zeros((hiddenSizes[layer - 1], hiddenSizes[layer])))

         self.hiddenBias.append(np.ones(hiddenSizes[layer]) * -1)
         self.updateBiasHidden.append(np.zeros(hiddenSizes[layer]))
         self.hiddenInput.append(np.zeros(hiddenSizes[layer]))
         self.hiddenNodes.append(np.zeros(hiddenSizes[layer]))


     self.outputWeights = (np.random.rand(hiddenSizes[nLayers - 1], outputSize) - 0.5)    
     self.outputBias = np.ones(outputSize) 
 
     self.updateWeightsOut = 0
     self.updateBiasOut = 0

     

   def resetBuffers(self):
     self.updateWeightsHidden = []
     self.updateBiasHidden = []
     for layer in range(self.nLayers):
         if layer == 0:
            self.updateWeightsHidden.append(np.zeros((self.nInputNodes, self.hiddenSizes[layer])))
         else:
            self.updateWeightsHidden.append(np.zeros((self.hiddenSizes[layer - 1], self.hiddenSizes[layer])))
         self.updateBiasHidden.append(np.zeros(self.hiddenSizes[layer]))
     self.updateWeightsOut = 0
     self.updateBiasOut = 0

   def getBrain(self):
     return [self.hiddenWeights, self.hiddenBias, self.outputWeights, self.outputBias]

   def setBrain(self, brain):
     self.hiddenWeights = brain[0]
     self.hiddenBias = brain[1]
     self.outputWeights = brain[2]
     self.outputBias = brain[3]
     self.resetBuffers()

   def activation(self, x):
     return self.tanh(x)

   def d_activation(self, act_x):
     return self.d_tanh(act_x)

   def dd_activation(self, inp):
     return self.dd_tanh(inp)

   def relu(self, x):
     return np.maximum(np.zeros(x.shape), x)
  
   def d_relu(self, relu_x):
     relu = relu_x
     relu[np.where(relu > 0)] = 1
     return relu

   def dd_relu(self, x):
     return np.zeros(x.shape)

   def sigmoid(self,x):
     return 1.0 / (1.0 + np.exp(-1. * x))
   
   def d_sigmoid(self,sig_x):
     return sig_x * (1.0 - sig_x)

   def dd_sigmoid(self, x):
     sig = self.sigmoid(x)
     return 1 - 2 * self.d_sigmoid(sig)

   def tanh(self, x):
     return np.tanh(x)

   def d_tanh(self, tanh_x):
     return 1 - tanh_x * tanh_x

   def dd_tanh(self, x):
     denom = np.cosh(2 * x) + 1
     denom = denom * denom * denom 
     return -1 * (8 * np.sinh(2 * x) * np.cosh(x) * np.cosh(x)) / (denom)



   def d_network(self):
     ret = (np.transpose(self.outputWeights) * self.d_activation(self.hiddenNodes)).dot(np.transpose(self.hiddenWeights))
     return [1, 1]
     return ret
     

   def dd_network(self):
     ret = (np.transpose(self.outputWeights) * self.dd_activation(self.hiddenInput)).dot(np.transpose(self.hiddenWeights * self.hiddenWeights))
     return [1, 1] 
     return ret

   def process(self, inputArray):
     for layer in range(self.nLayers):
        if layer == 0:
           self.hiddenInput[layer] = inputArray.dot(self.hiddenWeights[layer]) + self.hiddenBias[layer]
        else:
           self.hiddenInput[layer] = self.hiddenNodes[layer - 1].dot(self.hiddenWeights[layer]) + self.hiddenBias[layer]
        self.hiddenNodes[layer] = self.activation(self.hiddenInput[layer])
     self.outputNodes = self.hiddenNodes[self.nLayers - 1].dot(self.outputWeights) + self.outputBias  ##linear output layer
     return self.outputNodes
   
   def train(self, inputArray, targetOut, learningRate, momentum = 0):
     error = targetOut - self.process(inputArray)
     loss = 0.5 * error * error

     prevDeltaBiasHidden = self.updateBiasHidden
     prevDeltaWeightsHidden = self.updateWeightsHidden

     delta = (error)

     prevDeltaBiasOut = self.updateBiasOut
     self.updateBiasOut = delta
     
     prevDeltaWeightsOut = self.updateWeightsOut
     self.updateWeightsOut = np.transpose(np.outer(delta , self.hiddenNodes[self.nLayers - 1]))
     delta = delta.dot(np.transpose(self.outputWeights)) * (self.d_activation(self.hiddenNodes[self.nLayers - 1]))

     for idx in range(self.nLayers):
        layer = self.nLayers - idx - 1
        self.updateBiasHidden[layer] = delta
        
        if (layer == 0) :
           self.updateWeightsHidden[layer] = np.transpose(np.outer(delta, inputArray))
        else :
           self.updateWeightsHidden[layer] = np.transpose(np.outer(delta, self.hiddenNodes[layer - 1]))
           delta = delta.dot(np.transpose(self.hiddenWeights[layer])) * self.d_activation(self.hiddenNodes[layer - 1])




     self.outputWeights += learningRate * self.updateWeightsOut + momentum * prevDeltaWeightsOut
     self.outputBias += learningRate * self.updateBiasOut + momentum * prevDeltaBiasOut

     for idx in range(self.nLayers):
         layer = self.nLayers - idx - 1
         self.hiddenBias[layer] += learningRate * self.updateBiasHidden[layer] + momentum * prevDeltaBiasHidden[layer]
         self.hiddenWeights[layer] += learningRate * self.updateWeightsHidden[layer] + momentum * prevDeltaWeightsHidden[layer]

     return loss