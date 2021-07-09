import numpy as np

def conv(image, kernel, padding, strides=1):
    print("input\n",image)
    print(image.shape)
    print("filter\n",kernel)
    print(kernel.shape)
   
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    zKernShape = kernel.shape[2]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    zImgShape = image.shape[2]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((xImgShape + padding*2, yImgShape + padding*2, zImgShape))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image #slice, putting image in the correct position  
        print("imagepadded\n",imagePadded)
        print(imagePadded.shape)
    else:
        imagePadded = image
        
    # Iterate through image
    for z in range(zImgShape):
      for y in range(yImgShape):
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(xImgShape):
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                      if image[x][y][z]!=0:
                        output[x, y]= (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape, z: z + zKernShape]).sum()
                except:
                    break
    return output
  
x=128
y=128
z=3
input_3d = np.zeros((x, y,z))

i=0
j=0
for k in range(z):
 input_3d[i][j][k]=1


i=0
j=2
for k in range(z):
 input_3d[i][j][k]=1

i=2
j=1
for k in range(z):
 input_3d[i][j][k]=1

filter_3d = np.ones((3,3,z)) # 1 filer 
padding =int((filter_3d.shape[0]-1)/2)
output= conv(input_3d, filter_3d, padding, 1)
print("output\n",output)
print(output.shape)
  
  
