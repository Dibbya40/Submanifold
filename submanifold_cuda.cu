#include <stdlib.h>			
#include <iostream>
#include <math.h>
#include <time.h>
#include <sys/time.h>

timeval t1, t2;

__global__ void Submanifold_conv(float* image, float* filter, float* result, int image_Rows, int image_Cols, int filterRC, int filter_Depth, int result_Rows, int result_Cols, int padding)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;

	if (row < result_Rows && col < result_Cols)
	{       //printf("row %d col %d\n",row,col);
		int image_Row_Cols = image_Rows * image_Cols;

		for (int filterRow = 0; filterRow < filterRC; filterRow++) 
                {
			for (int filterCol = 0; filterCol < filterRC; filterCol++)
                        {
	                   for (int dep = 0; dep < filter_Depth; dep++)
                           {
                    
			    sum += image[(row + filterRow) * image_Cols + col + filterCol + dep * image_Row_Cols] * filter[filterRow * filterRC + filterCol + dep * filter_Depth];
                           }
			}
		}
                if(image[(row+padding) * (result_Cols+(padding*2)) + (col+padding)]!=0)
		result[row * result_Cols + col] = sum;
	}
}


void Convolution3D( float* image,  float* filter,  float* result, int padding, int image_Rows, int image_Cols, int filter_Rows, int filter_Depth,int result_Rows, int result_Cols)
{	int threadsPerBlock =32;

	int grid_Cols = ceil(float(result_Cols) / float(threadsPerBlock));
	int grid_Rows = ceil(float(result_Rows) / float(threadsPerBlock));

	dim3 gridDim(grid_Cols, grid_Rows);
	dim3 blockDim(threadsPerBlock,threadsPerBlock);		// total 32*32 = 1024 threads

        Submanifold_conv <<< gridDim, blockDim >>>(image,filter,result,image_Rows, image_Cols,filter_Rows, filter_Depth,result_Rows,result_Cols,padding );
}


int main() {

       	float *Mat1;//image
       	float *Mat2;//filter
        float *Mat3;//result
        float *padded_Mat1; 
	int filter_Size = 3;
        int padding=(filter_Size-1)/2;

	int Mat1_Rows = 4;
	int Mat1_Cols = 4;
	int Mat1_Depth = 3;
        int padded_Mat1_Rows = Mat1_Rows+(padding*2);
        int padded_Mat1_Cols = Mat1_Cols+(padding*2);
        int padded_Mat1_Depth = Mat1_Depth;

	int Mat2_Rows = filter_Size;
	int Mat2_Cols = filter_Size;
	int Mat2_Depth = 3;

	int Mat3_Rows = padded_Mat1_Rows - filter_Size + 1;
	int Mat3_Cols = padded_Mat1_Cols - filter_Size + 1;
	int Mat3_Depth = 1;

	int Mat1_Size = Mat1_Rows * Mat1_Cols * Mat1_Depth;
	int Mat2_Size = Mat2_Rows * Mat2_Cols * Mat2_Depth;
	int Mat3_Size = Mat3_Rows * Mat3_Cols * Mat3_Depth;
        int padded_Mat1_Size = padded_Mat1_Rows * padded_Mat1_Cols * padded_Mat1_Depth;
        
        //memory allocation
        cudaMallocManaged( & Mat1, Mat1_Size *sizeof(float)); //places variables in unified memory, available to CPU and GPU
        cudaMallocManaged( & Mat2, Mat2_Size *sizeof(float));
        cudaMallocManaged( & Mat3, Mat3_Size *sizeof(float));
        cudaMallocManaged( & padded_Mat1, padded_Mat1_Size *sizeof(float));
        int ii,jj,kk;

	for (int k = 0; k < padded_Mat1_Depth; k++)
        {

           for (int j=0; j < padded_Mat1_Cols; j++)
           {
	      for (int i=0; i < padded_Mat1_Rows; i++)
              {  
                   if((i==0 && j==0) || (i==2 && j ==0) || (i==1 && j==2))

                      { 
                           ii=i+padding;
                           jj=j+padding;
                           kk=k;    
                           padded_Mat1[(kk *padded_Mat1_Rows * padded_Mat1_Cols) + (jj * padded_Mat1_Rows) + ii]=1;
                      }
               
              }
           } 
        }
        
        for (size_t i = 0; i <Mat2_Size; i++)
           Mat2[i]=1;

        std::cout <<"padded_Mat1"<<std::endl;
        for (size_t dep = 0; dep < padded_Mat1_Depth; dep++)
        {  
          std::cout << std::endl;

          for (size_t col = 0; col < padded_Mat1_Cols; col++)
          {
                 for (size_t row = 0; row < padded_Mat1_Rows; row++)

                 {
                         std::cout << padded_Mat1[(dep *padded_Mat1_Rows * padded_Mat1_Cols) + (col * padded_Mat1_Rows) + row] << " ";
                 }
                 std::cout << std::endl;
          }
        }

        gettimeofday(&t1, 0); //time
	Convolution3D(padded_Mat1, Mat2, Mat3,padding,padded_Mat1_Rows,padded_Mat1_Cols, Mat2_Rows, Mat2_Depth,Mat3_Rows,Mat3_Cols);

	cudaDeviceSynchronize();
        gettimeofday(&t2, 0);//time
        double time = t2.tv_sec+(t2.tv_usec/1000000.0)- t1.tv_sec-(t1.tv_usec/1000000.0);

        printf("Convolution time:  %.6lf  s \n", time);

	for (size_t row = 0; row < Mat3_Rows; row++)
	{
		for (size_t col = 0; col < Mat3_Cols; col++)
		{
			std::cout << Mat3[row * Mat3_Cols + col] << " ";
		}
		std::cout << std::endl;
	}
	

	// cpu and gpu memory free
	cudaFree(Mat1);
	cudaFree(Mat2);
	cudaFree(Mat3);
        cudaFree(padded_Mat1);

	return 0;
}
