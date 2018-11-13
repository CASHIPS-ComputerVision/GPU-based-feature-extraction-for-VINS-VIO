#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>

#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace std;

__global__ static void test_kernel(unsigned char *img1,
				   const int row,
				   const int col
				  )
{
	const int tid=threadIdx.x;

	double scale=1.2f;

	for(int level=0;level<10;level++)
	{
		for(int i=tid;i<row/scale;i+=256)
		{
			float fy=(float)((i+0.5)*scale-0.5);
			int sy=(int)fy;
			fy-=sy;
			sy=(sy<row-2)?sy:row-2;
			sy=(sy<0)?0:sy;

			short cbufy[2];
			cbufy[0]=(short)((1.f-fy)*2048+0.5);
			cbufy[1]=2048-cbufy[0];

			for(int j=0;j<col/scale;j++)
			{
				float fx=(float)((j+ 0.5)*scale-0.5);
				int sx=(int)fx;
				fx-=sx;

				if(sx<0)
				{
					fx=0,sx=0;
				}
				if(sx>=col-1)
				{
					fx=0,sx=col-2;
				}

				short cbufx[2];
				cbufx[0]=(short)((1.f-fx)*2048+0.5);
				cbufx[1]=2048-cbufx[0];

				unsigned char temp=
					  (img1[sy*col+sx]*cbufx[0]*cbufy[0]+
					   img1[(sy+1)*col+1*sx]*cbufx[0]*cbufy[1]+
					   img1[sy*col+(sx+1)]*cbufx[1]*cbufy[0]+
					   img1[(sy+1)*col+(sx+1)]*cbufx[1]*cbufy[1])>>22;
			}
		}
		__syncthreads();
	}
}

void startgpu(cv::Mat image,unsigned char *img1_gpu)
{
	cudaMalloc((void**)&img1_gpu,sizeof(unsigned char)*image.rows*image.cols);

	cudaMemcpy(img1_gpu,image.data,sizeof(unsigned char)*image.rows*image.cols,cudaMemcpyHostToDevice);
}

void endgpu(unsigned char *img1_gpu)
{
	cudaFree(img1_gpu);
}

void addgpu(int img_row,int img_col,unsigned char *img1_gpu)
{
	test_kernel<<<1,1>>>(img1_gpu,img_row,img_col);
	cudaError_t error=cudaGetLastError();
}