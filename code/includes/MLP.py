import numpy as np


class MLP():
  
   nLayers = 2
   
   def __init__(self, nInputNodes, hiddenSize, outputSize):
     self.hiddenWeights = np.random.rand(nInputNodes, hiddenSize) * 0.001

     self.outputWeights = np.random.rand(hiddenSize, outputSize) * 0.001
     self.hiddenBias = np.ones(hiddenSize) * -1
     self.outputBias = np.ones(outputSize) * -1

     self.updateBiasHidden = 0
     self.updateWeightsHidden = 0
     self.updateWeightsOut = 0
     self.updateBiasOut = 0

     self.hiddenInput = 0
     pass

   def activation(self, x):
     #return self.sigmoid(x)
     return self.tanh(x)

   def d_activation(self, act_x):
     #return self.d_sigmoid(act_x)
     return self.d_tanh(act_x)

   def dd_activation(self, inp):
     #return self.dd_sigmoid(inp);
     return self.dd_tanh(inp)

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

   def d_hidden_layer(self):
     return self.d_sigmoid(self.hiddenNodes) * self.hiddenWeights


   def d_network(self):
     ##non linear outputlayer:
     ## d_out / d_x = d_out / d_hidden * d_hidden / d_x
     #ret = (self.d_activation(self.outputNodes).dot(np.transpose(self.outputWeights)) * self.d_activation(self.hiddenNodes)).dot(np.transpose(self.hiddenWeights))
     #return ret


     #linear output layer:
     ret = (np.transpose(self.outputWeights) * self.d_activation(self.hiddenNodes)).dot(np.transpose(self.hiddenWeights))
     #print "ret = " + str(ret)
     return ret
     

   def dd_network(self):
     ##non linear outputlayer:
     ##part 1 & 2 are the two parts of the product rule
     ## dd_out / dd_x = d/dx [(d_out / d_hidden) * (d_hidden/d_x)], which is a d/dx[f(x) * g(x)], hence product rule ->
     ## d/dx[f(x) * g(x)] = [(df(x) / dx) * g(x)] + [f(x) * (dg(x)/dx)]
     ## hence, 
     ## dd_out / dd_x = part1 + part2
     ## part1 = (d/dx[d_out / d_hidden] * [d_hidden/d_out] )
     ## part2 = [d_out/d_hidden] * d/dx[d_hidden/dx]


     #part1 = ((self.dd_activation(self.outputNodes)).dot(np.transpose(self.outputWeights * self.outputWeights)) * (self.d_activation(self.hiddenNodes) * self.d_activation(self.hiddenNodes))).dot(np.transpose((self.hiddenWeights * self.hiddenWeights)))
     #print "part1 " + str(part1)
     #part2 = ((self.d_activation(self.outputNodes)).dot(np.transpose(self.outputWeights)) * self.dd_activation(self.hiddenNodes)).dot(np.transpose(self.hiddenWeights * self.hiddenWeights))
     
     #print "part2" + str(part2)
     #return part1 + part2

     #linear output layer:
     ret = (np.transpose(self.outputWeights) * self.dd_activation(self.hiddenInput)).dot(np.transpose(self.hiddenWeights * self.hiddenWeights))
     
     return ret

   def process(self, inputArray):
     self.hiddenInput = inputArray.dot(self.hiddenWeights) + self.hiddenBias
     self.hiddenNodes = self.activation(self.hiddenInput)
     #self.outputNodes = self.activation(self.hiddenNodes.dot(self.outputWeights) + self.outputBias)
     self.outputNodes = self.hiddenNodes.dot(self.outputWeights) + self.outputBias  ##linear output layer
     #print self.outputNodes
     return self.outputNodes
   
   def train(self, inputArray, targetOut, learningRate, momentum):
     error = targetOut - self.process(inputArray)
     loss = 0.5 * error * error
     print loss

     #delta = (error) * self.d_activation(self.outputNodes) #non lin outlayer
     delta = (error) #* (self.outputNodes) #lin outlayer

     prevDeltaBiasOut = self.updateBiasOut
     self.updateBiasOut = delta
     
     prevDeltaWeightsOut = self.updateWeightsOut
     self.updateWeightsOut = np.transpose(np.outer(delta , self.hiddenNodes))

     delta = delta.dot(np.transpose(self.outputWeights)) * (self.d_activation(self.hiddenNodes))
     
     prevDeltaBiasHidden = self.updateBiasHidden
     self.updateBiasHidden = delta
     prevDeltaWeightsHidden = self.updateWeightsHidden
     self.updateWeightsHidden = np.transpose(np.outer(delta, inputArray))

     self.outputWeights += learningRate * self.updateWeightsOut + momentum * prevDeltaWeightsOut
     self.outputBias += learningRate * self.updateBiasOut + momentum * prevDeltaBiasOut
     self.hiddenBias += learningRate * self.updateBiasHidden + momentum * prevDeltaBiasHidden
     self.hiddenWeights += learningRate * self.updateWeightsHidden + momentum * prevDeltaWeightsHidden
     return loss
     
     
     
     
     
