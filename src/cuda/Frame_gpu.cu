#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
//#include <helper_cuda.h>
//#include <Utils.hpp>

//#include "Tracking.h"
//#include "Frame.h"
//#include "MapPoint.h"

//using namespace ORB_SLAM2;

const int isInFrustum_THREAD_NUM=256;

__global__ static void isInFrustum_kernel(const int sum,const float *mRcw,const float *mOw,
					  const float fx,const float cx,const float fy,const float cy,
					  const float mnMinX,const float mnMaxX,const float mnMinY,const float mnMaxY,
					  const float *minDistance,const float *maxDistance,
					  const float *P,const float *Pn,const float viewingCosLimit,
					  int *flag_gpu,float *u_gpu,float *v_gpu,
					  float *invz_gpu,float *dist_gpu,float *viewCos_gpu
					 )
{
    const int tid=threadIdx.x;
    const int bid=blockIdx.x;
    const int id=bid*blockDim.x+tid;
  
    if(id<sum)
    {
	int flag=0;
    
	// 3D in absolute coordinates
	const float PcX=mRcw[0]*P[id*3]+mRcw[1]*P[id*3+1]+mRcw[2]*P[id*3+2]+mRcw[3];
	const float PcY=mRcw[4]*P[id*3]+mRcw[5]*P[id*3+1]+mRcw[6]*P[id*3+2]+mRcw[7];
	const float PcZ=mRcw[8]*P[id*3]+mRcw[9]*P[id*3+1]+mRcw[10]*P[id*3+2]+mRcw[11];

	// Check positive depth
	if(PcZ<0.0f)
	    flag=1;

	// Project in image and check it is not outside
	const float invz = 1.0f/PcZ;
	const float u=fx*PcX*invz+cx;
	const float v=fy*PcY*invz+cy;
    
	if(u<mnMinX || u>mnMaxX || v<mnMinY || v>mnMaxY)
	   flag=2;

	// Check distance is in the scale invariance region of the MapPoint
	const double PO_0=P[id*3]-mOw[0];
	const double PO_1=P[id*3+1]-mOw[1];
	const double PO_2=P[id*3+2]-mOw[2];
	const float dist=sqrt(PO_0*PO_0+PO_1*PO_1+PO_2*PO_2);

	if(dist<minDistance[id]||dist>maxDistance[id])
	    flag=3;

	// Check viewing angle
	const float viewCos=(PO_0*Pn[id*3]+PO_1*Pn[id*3+1]+PO_2*Pn[id*3+2])/dist;

	if(viewCos<viewingCosLimit)
	    flag=4;

	flag_gpu[id]=flag;
	u_gpu[id]=u;
	v_gpu[id]=v;
	invz_gpu[id]=invz;
	dist_gpu[id]=dist;
	viewCos_gpu[id]=viewCos;
    }
}

void isInFrustum_gpu(const int sum,const float *mRcw,const float *mOw,
		     const float fx,const float cx,const float fy,const float cy,
		     const float mnMinX,const float mnMaxX,const float mnMinY,const float mnMaxY,
		     const float *minDistance,const float *maxDistance,
		     const float *P,const float *Pn,const float viewingCosLimit,
		     int *flag,float *u,float *v,
		     float *invz,float *dist,float *viewCos
		    )
{
  
    float *mRcw_GPU,*mOw_GPU,*minDistance_GPU,*maxDistance_GPU,*P_GPU,*Pn_GPU,*u_gpu,*v_gpu,*invz_gpu,*dist_gpu,*viewCos_GPU;
    int *flag_gpu;

    cudaMalloc((void**)&mRcw_GPU,sizeof(float)*12);
    cudaMalloc((void**)&mOw_GPU,sizeof(float)*3);
    cudaMalloc((void**)&minDistance_GPU,sizeof(float)*sum);
    cudaMalloc((void**)&maxDistance_GPU,sizeof(float)*sum);
    cudaMalloc((void**)&P_GPU,sizeof(float)*sum*3);
    cudaMalloc((void**)&Pn_GPU,sizeof(float)*sum*3);
    cudaMalloc((void**)&flag_gpu,sizeof(int)*sum);
    cudaMalloc((void**)&u_gpu,sizeof(float)*sum);
    cudaMalloc((void**)&v_gpu,sizeof(float)*sum);
    cudaMalloc((void**)&invz_gpu,sizeof(float)*sum);
    cudaMalloc((void**)&dist_gpu,sizeof(float)*sum);
    cudaMalloc((void**)&viewCos_GPU,sizeof(float)*sum);
    
    cudaMemcpy(mRcw_GPU,mRcw,sizeof(float)*12,cudaMemcpyHostToDevice);
    cudaMemcpy(mOw_GPU,mOw,sizeof(float)*3,cudaMemcpyHostToDevice);
    cudaMemcpy(minDistance_GPU,minDistance,sizeof(float)*sum,cudaMemcpyHostToDevice);
    cudaMemcpy(maxDistance_GPU,maxDistance,sizeof(float)*sum,cudaMemcpyHostToDevice);
    cudaMemcpy(P_GPU,P,sizeof(float)*sum*3,cudaMemcpyHostToDevice);
    cudaMemcpy(Pn_GPU,Pn,sizeof(float)*sum*3,cudaMemcpyHostToDevice);
    
    isInFrustum_kernel<<<ceil((float)sum/isInFrustum_THREAD_NUM),isInFrustum_THREAD_NUM>>>(sum,mRcw_GPU,mOw_GPU,
				fx,cx,fy,cy,
				mnMinX,mnMaxX,mnMinY,mnMaxY,
				minDistance_GPU,maxDistance_GPU,
				P_GPU,Pn_GPU,viewingCosLimit,
				flag_gpu,u_gpu,v_gpu,
				invz_gpu,dist_gpu,viewCos_GPU);
    cudaError_t error=cudaGetLastError();
    std::cout<<"isInFrustum_kernel:"<<cudaGetErrorString(error)<<std::endl;
    
    cudaMemcpy(flag,flag_gpu,sizeof(int)*sum,cudaMemcpyDeviceToHost);
    cudaMemcpy(u,u_gpu,sizeof(float)*sum,cudaMemcpyDeviceToHost);
    cudaMemcpy(v,v_gpu,sizeof(float)*sum,cudaMemcpyDeviceToHost);
    cudaMemcpy(invz,invz_gpu,sizeof(float)*sum,cudaMemcpyDeviceToHost);
    cudaMemcpy(dist,dist_gpu,sizeof(float)*sum,cudaMemcpyDeviceToHost);
    cudaMemcpy(viewCos,viewCos_GPU,sizeof(float)*sum,cudaMemcpyDeviceToHost);
    
    cudaFree(mRcw_GPU);
    cudaFree(mOw_GPU);
    cudaFree(minDistance_GPU);
    cudaFree(maxDistance_GPU);
    cudaFree(P_GPU);
    cudaFree(Pn_GPU);
    cudaFree(flag_gpu);
    cudaFree(u_gpu);
    cudaFree(v_gpu);
    cudaFree(invz_gpu);
    cudaFree(dist_gpu);
    cudaFree(viewCos_GPU);
}