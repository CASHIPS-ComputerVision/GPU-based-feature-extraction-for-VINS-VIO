#pragma once
#ifndef __ORB_HPP__
#define __ORB_HPP__

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

namespace ORB_SLAM2 { namespace cuda {
  using namespace std;
  using namespace cv;
  using namespace cv::cuda;

  class GpuOrb {
    unsigned int maxKeypoints;
    KeyPoint * keypoints[3];
    GpuMat descriptors[3];
    GpuMat desc[3];
    cudaStream_t stream[3];
    Stream cvStream[3];
  public:
    GpuOrb(int maxKeypoints = 10000);
    ~GpuOrb();

    void launch_async(InputArray _image, const KeyPoint * _keypoints, const int npoints,vector<KeyPoint*> keypoints_mul_GPU,float scale,int c);
    void launch_async_mul(std::vector<cv::cuda::GpuMat> _images, vector<vector<KeyPoint> > *allKeypoints, int level);
    void join(Mat &_descriptors,vector<KeyPoint> &_keypoints,vector<KeyPoint*> keypoints_mul_GPU,int c);

    static void loadPattern(const Point * _pattern);
  };
} }
#endif
