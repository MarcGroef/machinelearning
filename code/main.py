from includes.MLP import MLP
import numpy as np


if __name__ == "__main__":
  
  xorIn1 = np.array([0,1])
  xorIn2 = np.array([1,0])
  xorIn3 = np.array([1,1])
  xorIn4 = np.array([0,0])
  
  xorOut1 = np.array([1])
  xorOut2 = np.array([1])
  xorOut3 = np.array([0])
  xorOut4 = np.array([0])
  
  
  nn = MLP(2, 20, 1)
  for iter in range(2500):
    loss = 0
    loss += nn.train(xorIn1, xorOut1, 0.5)
    loss += nn.train(xorIn4, xorOut4, 0.5)
    loss += nn.train(xorIn2, xorOut2, 0.5)
    loss += nn.train(xorIn3, xorOut3, 0.5)
    
    print loss
  