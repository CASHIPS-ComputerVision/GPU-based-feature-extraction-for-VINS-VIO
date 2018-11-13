#pragma once
#ifndef __FAST_HPP__
#define __FAST_HPP__

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

namespace ORB_SLAM2 { namespace cuda {
  using namespace std;
  using namespace cv;
  using namespace cv::cuda;
  
  #define CAMS 3

  const float FEATURE_SIZE = 7.0;

  class GpuFast {
    short2 * kpLoc;
    float * kpScore;
    unsigned int * counter_ptr;
    unsigned int highThreshold;
    unsigned int lowThreshold;
    unsigned int maxKeypoints;
    unsigned int count[3];
    cv::cuda::GpuMat scoreMat[3];
    cv::cuda::GpuMat scoreMat_mul[3][8];
    cudaStream_t stream[3];
    Stream cvStream[3];
  public:
    GpuFast(int highThreshold, int lowThreshold, int maxKeypoints = 10000);
    ~GpuFast();

    void detect(InputArray, std::vector<KeyPoint>&);

    void detectAsync(InputArray,int c=0);
    void detectAsync_mul(InputArray,InputArray,InputArray,int);
    void joinDetectAsync(std::vector<KeyPoint>&);
    void joinDetectAsync_mul(std::vector<KeyPoint> *,InputArray,InputArray,InputArray);
  };

  class IC_Angle {
    unsigned int maxKeypoints;
    KeyPoint * keypoints[3];
    KeyPoint * keypoints_mul[3][8];
    cudaStream_t stream[3];
    Stream _cvStream[3];
  public:
    IC_Angle(unsigned int maxKeypoints = 10000);
    ~IC_Angle();
    void launch_async_mul(std::vector<cv::cuda::GpuMat> _images,vector<vector<KeyPoint> > *_keypoints,vector<vector<KeyPoint*> > &keypoints_mul_GPU,int half_k, int minBorderX, int minBorderY, int octave, int size);

    Stream& cvStream(int c) { return _cvStream[c];}
    static void loadUMax(const int* u_max, int count);
  };
} }
#endif
