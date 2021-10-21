import numpy as np

def active_site( i, j, z, input_3d):
    for k in range(z):
      input_3d[i][j][k]=1
    return input_3d

def rulebook(input_3d):
    l_x=[]
    l_y=[]
    x_shape = input_3d.shape[0]
    y_shape = input_3d.shape[1]
    for i in range(x_shape):
     for j in range(y_shape):
       if input_3d[i][j][0]!=0:
          l_x.append(i)
          l_y.append(j)    

    return l_x,l_y
class ConvolutionFunction(Function):
  @staticmethod
  def forward(ctx,image, kernel, padding,l_x,l_y,strides=1):
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


    for z in range(zImgShape):
      for i in range(len(l_y)):
        try:
         output[l_x[i],l_y[i]]= (kernel * imagePadded[l_x[i]: l_x[i] + xKernShape, l_y[i]: l_y[i] + yKernShape, z: z + zKernShape]).sum()
        except:
         break
    ctx.save_for_backward(image,kernel)    
    return output
  
  @staticmethod
  def backward(ctx, grad_output):
    (image,kernel)=ctx.saved_tensors
    
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    zKernShape = kernel.shape[2]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    zImgShape = image.shape[2]
    
    d_imagePadded = np.zeros(imagePadded.shape)
    d_kernel = np.zeros(kernel.shape)
    
    for z in range(zImgShape):
      for i in range(len(l_y)):
        try:
         d_imagePadded[l_x[i]: l_x[i] + xKernShape, l_y[i]: l_y[i] + yKernShape, z: z + zKernShape]+=kernel*grad_output(l_x[i],l_y[i],z)
         d_kernel+=imagePadded[l_x[i]: l_x[i] + xKernShape, l_y[i]: l_y[i] + yKernShape, z: z + zKernShape]*grad_output(l_x[i],l_y[i],z)
        except:
         break
    return d_imagePadded,d_kernel
  
x=640
y=480
z=3
input_3d = np.zeros((x, y,z))

for a in range(50):
  if a%5==0:
    for b in range(50):
      if b%7==0:
        inp_3d=active_site(a, b, z, input_3d) #(0,0),(0,7)..(0,49), (5,0),(5,7)..(5,49), (10,0)...(10,49)   ...... (45,49)/total 80 active

'''
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
'''

filter_3d = np.ones((3,3,z)) # 1 filer 
padding =int((filter_3d.shape[0]-1)/2)
l_x,l_y=rulebook(input_3d)
output= conv(inp_3d, filter_3d, padding,l_x,l_y,1)
print("output\n",output)
print(output.shape)
#print(nActive)
