import torch, data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys
import math
import numpy as np

data.init(-1,24,24*8+15,16)
dimension = 3
reps = 2 #Conv block repetition factor
m = 32 #Unet number of features
nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, data.spatialSize, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(
           scn.FullyConvolutionalNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[3,2])).add(
           scn.BatchNormReLU(sum(nPlanes))).add(
           scn.OutputLayer(dimension))
        self.linear = nn.Linear(sum(nPlanes), data.nClassesTotal)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        return x

model=Model()
print(model)
trainIterator=data.train()
validIterator=data.valid()      
      
