/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include <helper_cuda.h>
#include <cuda/Orb.hpp>
#include <Utils.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace cv::cuda::device;

namespace ORB_SLAM2 { namespace cuda {

  __constant__ unsigned char c_pattern[sizeof(Point) * 512];

  void GpuOrb::loadPattern(const Point * _pattern) {
    checkCudaErrors( cudaMemcpyToSymbol(c_pattern, _pattern, sizeof(Point) * 512) );
  }

#define GET_VALUE(idx) \
    image(loc.y + __float2int_rn(pattern[idx].x * b + pattern[idx].y * a), \
          loc.x + __float2int_rn(pattern[idx].x * a - pattern[idx].y * b))

  __global__ void calcOrb_kernel(const PtrStepb image, KeyPoint * keypoints, const int npoints, PtrStepb descriptors,const int scale) {
    int id = blockIdx.x;
    int tid = threadIdx.x;
    if (id >= npoints) return;

    const KeyPoint &kpt = keypoints[id];
    short2 loc = make_short2(kpt.pt.x, kpt.pt.y);
    const Point * pattern = ((Point *)c_pattern) + 16 * tid;

    uchar * desc = descriptors.ptr(id);
    const float factorPI = (float)(CV_PI/180.f);
    float angle = (float)kpt.angle * factorPI;
    float a = (float)cosf(angle), b = (float)sinf(angle);

    int t0, t1, val;
    t0 = GET_VALUE(0); t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2); t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4); t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6); t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8); t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10); t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12); t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14); t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    desc[tid] = (uchar)val;
  }
  
  __global__ void calcOrb_kernel_mul(const PtrStepb image1,const PtrStepb image2,const PtrStepb image3,
				     KeyPoint * keypoints1,KeyPoint * keypoints2,KeyPoint * keypoints3,
				     const int npoints1,const int npoints2,const int npoints3,
				     PtrStepb descriptors1,PtrStepb descriptors2,PtrStepb descriptors3)
  {
    int id = blockIdx.x;
    int tid = threadIdx.x;
    int c = blockIdx.y;
    
    PtrStepb image;
    KeyPoint * keypoints;
    int npoints;
    PtrStepb descriptors;
    
    if(c==0)
    {
      image=image1;
      keypoints=keypoints1;
      npoints=npoints1;
      descriptors=descriptors1;
    }
    if(c==1)
    {
      image=image2;
      keypoints=keypoints2;
      npoints=npoints2;
      descriptors=descriptors2;
    }
    if(c==2)
    {
      image=image3;
      keypoints=keypoints3;
      npoints=npoints3;
      descriptors=descriptors3;
    }
    
    if (id >= npoints) return;

    const KeyPoint &kpt = keypoints[id];
    short2 loc = make_short2(kpt.pt.x, kpt.pt.y);
    const Point * pattern = ((Point *)c_pattern) + 16 * tid;

    uchar * desc = descriptors.ptr(id);
    const float factorPI = (float)(CV_PI/180.f);
    float angle = (float)kpt.angle * factorPI;
    float a = (float)cosf(angle), b = (float)sinf(angle);

    int t0, t1, val;
    t0 = GET_VALUE(0); t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2); t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4); t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6); t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8); t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10); t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12); t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14); t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    desc[tid] = (uchar)val;
  }

#undef GET_VALUE

__global__ void changeScale_kernel(KeyPoint * keypoints,const int npoints,const float scale) {
    int tid = threadIdx.x;
    if (tid >= npoints) {
      return;
    }
    keypoints[tid].pt.x *= scale;
    keypoints[tid].pt.y *= scale;
  }

  GpuOrb::GpuOrb(int maxKeypoints) : maxKeypoints(maxKeypoints)
  {
    checkCudaErrors( cudaStreamCreate(&stream[0]) );
    checkCudaErrors( cudaStreamCreate(&stream[1]) );
    checkCudaErrors( cudaStreamCreate(&stream[2]) );
    
    cvStream[0] = StreamAccessor::wrapStream(stream[0]);
    cvStream[1] = StreamAccessor::wrapStream(stream[1]);
    cvStream[2] = StreamAccessor::wrapStream(stream[2]);
    
    checkCudaErrors( cudaMalloc(&keypoints[0], sizeof(KeyPoint) * maxKeypoints) );
    checkCudaErrors( cudaMalloc(&keypoints[1], sizeof(KeyPoint) * maxKeypoints) );
    checkCudaErrors( cudaMalloc(&keypoints[2], sizeof(KeyPoint) * maxKeypoints) );
    
    descriptors[0]=GpuMat(maxKeypoints, 32, CV_8UC1);
    descriptors[1]=GpuMat(maxKeypoints, 32, CV_8UC1);
    descriptors[2]=GpuMat(maxKeypoints, 32, CV_8UC1);
  }

  GpuOrb::~GpuOrb() {
    cvStream[0].~Stream();
    cvStream[1].~Stream();
    cvStream[2].~Stream();
    
    checkCudaErrors( cudaFree(keypoints[0]) );
    checkCudaErrors( cudaFree(keypoints[1]) );
    checkCudaErrors( cudaFree(keypoints[2]) );
    
    checkCudaErrors( cudaStreamDestroy(stream[0]) );
    checkCudaErrors( cudaStreamDestroy(stream[1]) );
    checkCudaErrors( cudaStreamDestroy(stream[2]) );
  }

  void GpuOrb::launch_async(InputArray _image, const KeyPoint * _keypoints, const int npoints,vector<KeyPoint*> keypoints_mul_GPU,float scale,int c) {
    if (npoints == 0) {
      POP_RANGE;
      return ;
    }
    const GpuMat image = _image.getGpuMat();

    desc[c] = descriptors[c].rowRange(0, npoints);
    desc[c].setTo(Scalar::all(0), cvStream[c]);

    dim3 dimBlock(32);
    dim3 dimGrid(npoints);
    calcOrb_kernel<<<dimGrid, dimBlock, 0, stream[c]>>>(image.rowRange(image.rows/3*c, image.rows/3*(c+1)), keypoints_mul_GPU[c], npoints, desc[c],scale);
    changeScale_kernel<<<1, npoints, 0, stream[c]>>>(keypoints_mul_GPU[c],npoints,scale);
    checkCudaErrors( cudaGetLastError() );
  }
  
  void GpuOrb::launch_async_mul(std::vector<cv::cuda::GpuMat> _images, vector<vector<KeyPoint> > *allKeypoints, int level) {
    int npoints[3];
    int npoint=0;
    for(int c=0;c<3;c++){
    if ((npoints[c]=allKeypoints[c][level].size())==0) {
      continue;
    }

    checkCudaErrors( cudaMemcpyAsync(keypoints[c], allKeypoints[c][level].data(), sizeof(KeyPoint) * npoints[c], cudaMemcpyHostToDevice, stream[c]) );
    desc[c] = descriptors[c].rowRange(0, npoints[c]);
    desc[c].setTo(Scalar::all(0), cvStream[c]);
    if(npoints[c]>npoint)npoint=npoints[c];
    }

    dim3 dimBlock(32);
      if (npoint==0) {
      POP_RANGE;
      return ;
    }
    dim3 dimGrid(npoint,3);
    calcOrb_kernel_mul<<<dimGrid, dimBlock, 0, stream[0]>>>(_images[level*3+0],_images[level*3+1],_images[level*3+2],
							    keypoints[0],keypoints[1],keypoints[2],
							    npoints[0],npoints[1],npoints[2],
							    desc[0],desc[1],desc[2]);
    checkCudaErrors( cudaGetLastError() );
  }

  void GpuOrb::join(Mat &_descriptors,vector<KeyPoint> &_keypoints,vector<KeyPoint*> keypoints_mul_GPU,int c) {
    desc[c].download(_descriptors, cvStream[c]);
    checkCudaErrors( cudaMemcpyAsync(_keypoints.data(), keypoints_mul_GPU[c], sizeof(KeyPoint) * _keypoints.size(), cudaMemcpyDeviceToHost, stream[c]) );
    checkCudaErrors( cudaStreamSynchronize(stream[c]) );
  }
} }
