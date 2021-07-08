import numpy as np
import scipy
from scipy import ndimage
import scipy.signal as sps
from scipy import signal
x=4
y=4
z=3
input_3d = np.zeros((x, y, z))
padded_input_3d=np.zeros((x+2,y+2,z))
i=0
j=0
for k in range(3):
  input_3d[i][j][k]=1
  padded_input_3d[i+1][j+1][k]=1

i=0
j=2
for k in range(3):
  input_3d[i][j][k]=1
  padded_input_3d[i+1][j+1][k]=1

i=2
j=1
for k in range(3):
  input_3d[i][j][k]=1
  padded_input_3d[i+1][j+1][k]=1

filter_3d = np.ones((3,3,3)) # 1 filer 

output_2d=signal.convolve(padded_input_3d,filter_3d, mode='valid')

print("input_3d",input_3d)
#print(input_3d.shape)

#submanifold operation: if input_3d[0 , 2 , (0,1,2) ]=1 , output_2d[0 , 2 , (0)]=1
z1=0
for i in range(0, x):
 for j in range(0, y):
  if input_3d[i][j][z1]!=1:
     output_2d[i][j][z1]=0 # z1=0, it is 2d 

print("output_2d",output_2d)
#print(output_2d.shape) #z=1, it is 2d
print("filter_3d",filter_3d)
#print(filter_3d.shape)

