/**
* This file is part of MultiCol-SLAM
*
* Copyright (C) 2015-2016 Steffen Urban <urbste at googlemail.com>
* For more information see <https://github.com/urbste/MultiCol-SLAM>
*
* MultiCol-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* MultiCol-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with MultiCol-SLAM . If not, see <http://www.gnu.org/licenses/>.
*/

/*
* MultiCol-SLAM is based on ORB-SLAM2 which was also released under GPLv3
* For more information see <https://github.com/raulmur/ORB_SLAM2>
* Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
*/


/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*********************************************************************/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iterator>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include "mdBRIEFextractorOct.h"
#include "cuda/Allocator.hpp"
#include "cuda/Fast.hpp"
#include "cuda/Orb.hpp"

#include "misc.h"
#include "Utils.hpp"

namespace MultiColSLAM
{

using namespace cv;
using namespace std;

const float HARRIS_K = 0.04f;
const float DEG2RADf = static_cast<float>(CV_PI) / 180.f;
const int PATCH_SIZE = 32;
const int HALF_PATCH_SIZE = 16;
const int EDGE_THRESHOLD = 25;

static void HarrisResponses(const Mat& img,
	const std::vector<Rect>& layerinfo,
	std::vector<KeyPoint>& pts,
	int blockSize,
	float harris_k)
{
	CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);

	size_t ptidx, ptsize = pts.size();

	const uchar* ptr00 = img.ptr<uchar>();
	int step = (int)(img.step / img.elemSize1());
	int r = blockSize / 2;

	float scale = 1.f / ((1 << 2) * blockSize * 255.f);
	float scale_sq_sq = scale * scale * scale * scale;

	AutoBuffer<int> ofsbuf(blockSize*blockSize);
	int* ofs = ofsbuf;
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			ofs[i*blockSize + j] = (int)(i*step + j);

	for (ptidx = 0; ptidx < ptsize; ptidx++)
	{
		int x0 = cvRound(pts[ptidx].pt.x);
		int y0 = cvRound(pts[ptidx].pt.y);
		int z = pts[ptidx].octave;

		const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
		int a = 0, b = 0, c = 0;

		for (int k = 0; k < blockSize*blockSize; k++)
		{
			const uchar* ptr = ptr0 + ofs[k];
			int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
			int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
			a += Ix*Ix;
			b += Iy*Iy;
			c += Ix*Iy;
		}
		pts[ptidx].response = ((float)a * b - (float)c * c -
			harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
	}
}

mdBRIEFextractorOct::mdBRIEFextractorOct(int _nfeatures,
	float _scaleFactor,
	int _nlevels,
	int _edgeThreshold,
	int _firstLevel,
	int _scoreType,
	int _patchSize,
	int _fastThreshold,
	int _minThreshold,
	bool _useAgast,
	int _fastAgastType,
	bool _do_dBrief,
	bool _learnMasks,
	int _descSize) :
	nfeatures(_nfeatures), scaleFactor(_scaleFactor), numlevels(_nlevels),
	edgeThreshold(_edgeThreshold), firstLevel(_firstLevel),scoreType(_scoreType),
	patchSize(_patchSize), fastThreshold(_fastThreshold),minThreshold(_minThreshold),
	useAgast(_useAgast), fastAgastType(_fastAgastType), learnMasks(_learnMasks),
	descSize(_descSize), do_dBrief(_do_dBrief),
	gpuFast(fastThreshold, minThreshold)
{
	mvScaleFactor.resize(numlevels);
	mvScaleFactor[0] = 1;
	for (int i = 1; i < numlevels; i++)
		mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;

	double invScaleFactor = 1.0 / scaleFactor;
	mvInvScaleFactor.resize(numlevels);
	mvInvScaleFactor[0] = 1;
	for (int i = 1; i < numlevels; i++)
		mvInvScaleFactor[i] = mvInvScaleFactor[i - 1] * invScaleFactor;

	// Postpone the allocation of the Pyramids to the time we process the first frame.
	mvImagePyramidAllocatedFlag = false;

	mvMaskPyramid.resize(numlevels*CAMS);
	
	mvMultiImagePyramid.resize(numlevels);
	mvMultiImagePyramidBorder.resize(numlevels);

	mnFeaturesPerLevel.resize(numlevels);
	double factor = (1.0 / scaleFactor);
	double nDesiredFeaturesPerScale = nfeatures*(1 - factor) /
		(1 - pow(factor, numlevels));

	int sumFeatures = 0;
	for (int level = 0; level < numlevels - 1; level++)
	{
		mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
		sumFeatures += mnFeaturesPerLevel[level];
		nDesiredFeaturesPerScale *= factor;
	}
	mnFeaturesPerLevel[numlevels - 1] = std::max(nfeatures - sumFeatures, 0);

	const int npoints = 2 * 8 * descSize;
	const Point* pattern0 = (const Point*)learned_pattern_64_ORB;
	std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

	//This is for orientation
	// pre-compute the end of a row in a circular patch
	umax.resize(HALF_PATCH_SIZE + 1);

	int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
	int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
	const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
	for (v = 0; v <= vmax; ++v)
		umax[v] = cvRound(sqrt(hp2 - v * v));

	// Make sure we are symmetric
	for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
	{
		while (umax[v0] == umax[v0 + 1])
			++v0;
		umax[v] = v0;
		++v0;
	}
	
	ORB_SLAM2::cuda::IC_Angle::loadUMax(umax.data(), umax.size());
	ORB_SLAM2::cuda::GpuOrb::loadPattern(pattern.data());
}


int mdBRIEFextractorOct::descriptorSize() const
{
	return descSize;
}

int mdBRIEFextractorOct::descriptorType() const
{
	return CV_8U;
}

int mdBRIEFextractorOct::defaultNorm() const
{
	return NORM_HAMMING;
}

static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

static void rotateAndDistortPattern(const Point2d& undist_kps,
	const std::vector<Point>& patternIn,
	std::vector<Point>& patternOut,
	cCamModelGeneral_& camModel,
	const double& ax,
	const double& ay)
{
	const size_t npoints = patternIn.size();
	const double npointsd = static_cast<double>(npoints);
	std::vector<double> xcoords(npoints);
	std::vector<double> ycoords(npoints);
	double sumX = 0.0;
	double sumY = 0.0;

	for (size_t p = 0; p < npoints; ++p)
	{
		// rotate pattern point and move it to the keypoint
		double xr = patternIn[p].x*ax - patternIn[p].y*ay + undist_kps.x;
		double yr = patternIn[p].x*ay + patternIn[p].y*ax + undist_kps.y;

		camModel.distortPointsOcam(xr, yr, xcoords[p], ycoords[p]);

		sumX += xcoords[p];
		sumY += ycoords[p];
	}
	double meanX = sumX / npointsd;
	double meanY = sumY / npointsd;
	// substract mean, to get correct pattern size
	for (size_t p = 0; p < npoints; ++p)
	{
		patternOut[p].x = cvRound(xcoords[p] - meanX);
		patternOut[p].y = cvRound(ycoords[p] - meanY);
	}
}

static void rotatePattern(
	const std::vector<Point>& patternIn,
	std::vector<Point>& patternOut,
	const double& ax,
	const double& ay)
{
	const int npoints = (int)patternIn.size();
	for (int p = 0; p < npoints; ++p)
	{
		// rotate pattern point
		patternOut[p].x = cvRound(patternIn[p].x*ax - patternIn[p].y*ay);
		patternOut[p].y = cvRound(patternIn[p].x*ay + patternIn[p].y*ax);
		++p;
		patternOut[p].x = cvRound(patternIn[p].x*ax - patternIn[p].y*ay);
		patternOut[p].y = cvRound(patternIn[p].x*ay + patternIn[p].y*ax);
	}
}

static void compute_ORB(const Mat& image,
	const KeyPoint& keypoint,
	const std::vector<Point>& _pattern,
	cCamModelGeneral_& camModel,
	uchar* descriptor,
	const int& descsize)
{
	const int npoints = _pattern.size();

	std::vector<Point> rotatedPattern(npoints);
	double angle = static_cast<double>(keypoint.angle*DEG2RADf);

	rotatePattern(_pattern,
		rotatedPattern, cos(angle), sin(angle));

	const Point* pattern = &rotatedPattern[0];
	const uchar* center = 0;

	int ix = 0, iy = 0;

	int row = cvRound(keypoint.pt.y);
	int col = cvRound(keypoint.pt.x);
#define GET_VALUE(idx) \
               (ix = pattern[idx].x, \
                iy = pattern[idx].y, \
				center = image.ptr<uchar>(row+iy),\
                center[col+ix] )

	for (int i = 0; i < descsize; ++i, pattern += 16)
	{
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

		descriptor[i] = (uchar)val;
	}
#undef GET_VALUE
}

static void compute_dBRIEF(const Mat& image,
	const KeyPoint& keypoint,
	const Vec2d& undistortedKeypoint,
	const std::vector<Point>& _pattern,
	cCamModelGeneral_& camModel,
	uchar* descriptor,
	const int& descsize)
{
	const int npoints = _pattern.size();

	std::vector<Point> distortedRotatedPattern(npoints);
	double angle = static_cast<double>(keypoint.angle*DEG2RADf);

	rotateAndDistortPattern(undistortedKeypoint, _pattern,
		distortedRotatedPattern, camModel, cos(angle), sin(angle));

	const Point* pattern = &distortedRotatedPattern[0];
	const uchar* center = 0;

	int ix = 0, iy = 0;

	int row = cvRound(keypoint.pt.y);
	int col = cvRound(keypoint.pt.x);
#define GET_VALUE(idx) \
               (ix = pattern[idx].x, \
                iy = pattern[idx].y, \
				center = image.ptr<uchar>(row+iy),\
                center[col+ix] )

	for (int i = 0; i < descsize; ++i, pattern += 16)
	{
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

		descriptor[i] = (uchar)val;
	}
#undef GET_VALUE
}

static void compute_mdBRIEF(const Mat& image,
	const KeyPoint& keypoint,
	const Vec2d& undistortedKeypoint,
	const std::vector<Point>& _pattern,
	cCamModelGeneral_& camModel,
	uchar* descriptor,
	uchar* descMask,
	const int& descsize)
{
	const int npoints = 2 * 8 * descsize; //2*512

	std::vector<Point> distortedRotatedPattern(npoints);
	std::vector<std::vector<Point>> maskPattern(2, std::vector<Point>(npoints));
	// the two rotations to learn the mask
	double rot = 20.0 / RHOd;
	double angle = static_cast<double>(keypoint.angle / RHOf);
	double angle1 = angle + rot;
	double angle2 = angle - rot;

	rotateAndDistortPattern(undistortedKeypoint, _pattern,
		distortedRotatedPattern, camModel, cos(angle), sin(angle));

	rotateAndDistortPattern(undistortedKeypoint, _pattern,
		maskPattern[0], camModel, cos(angle1), sin(angle1));

	rotateAndDistortPattern(undistortedKeypoint, _pattern,
		maskPattern[1], camModel, cos(angle2), sin(angle2));

	const Point* pattern = &distortedRotatedPattern[0];
	const Point* maskPattern1 = &maskPattern[0][0];
	const Point* maskPattern2 = &maskPattern[1][0];
	const uchar* center = 0;

	int row = cvRound(keypoint.pt.y);
	int col = cvRound(keypoint.pt.x);
	int ix = 0, iy = 0;
#define GET_VALUE(idx) \
               (ix = pattern[idx].x, \
                iy = pattern[idx].y, \
				center = image.ptr<uchar>(row+iy),\
                center[col+ix] )
#define GET_VALUE_MASK1(idx) \
				(ix = maskPattern1[idx].x, \
                 iy = maskPattern1[idx].y, \
				center = image.ptr<uchar>(row+iy),\
                center[col+ix] )
#define GET_VALUE_MASK2(idx) \
				(ix = maskPattern2[idx].x, \
                 iy = maskPattern2[idx].y, \
				center = image.ptr<uchar>(row+iy),\
                center[col+ix] )
	for (int i = 0; i < descsize; ++i, pattern += 16,
		maskPattern1 += 16, maskPattern2 += 16)
	{
		int temp_val;
		int t0, t1, val, maskVal;
		int mask1_1, mask1_2, mask2_1, mask2_2, stable_val = 0;
		// first bit
		t0 = GET_VALUE(0); t1 = GET_VALUE(1);
		temp_val = t0 < t1;
		val = temp_val;
		mask1_1 = GET_VALUE_MASK1(0); mask1_2 = GET_VALUE_MASK1(1); // mask1
		mask2_1 = GET_VALUE_MASK2(0); mask2_2 = GET_VALUE_MASK2(1); // mask2
		stable_val += (mask1_1 < mask1_2) ^ temp_val;
		stable_val += (mask2_1 < mask2_2) ^ temp_val;
		maskVal = (stable_val == 0);
		stable_val = 0;
		// second bit
		t0 = GET_VALUE(2); t1 = GET_VALUE(3);
		temp_val = t0 < t1;
		val |= temp_val << 1;
		mask1_1 = GET_VALUE_MASK1(2); mask1_2 = GET_VALUE_MASK1(3); // mask1
		mask2_1 = GET_VALUE_MASK2(2); mask2_2 = GET_VALUE_MASK2(3); // mask2
		stable_val += (mask1_1 < mask1_2) ^ temp_val;
		stable_val += (mask2_1 < mask2_2) ^ temp_val;
		maskVal |= (stable_val == 0) << 1;
		stable_val = 0;
		// third bit
		t0 = GET_VALUE(4); t1 = GET_VALUE(5);
		temp_val = t0 < t1;
		val |= temp_val << 2;
		mask1_1 = GET_VALUE_MASK1(4); mask1_2 = GET_VALUE_MASK1(5); // mask1
		mask2_1 = GET_VALUE_MASK2(4); mask2_2 = GET_VALUE_MASK2(5); // mask2
		stable_val += (mask1_1 < mask1_2) ^ temp_val;
		stable_val += (mask2_1 < mask2_2) ^ temp_val;
		maskVal |= (stable_val == 0) << 2;
		stable_val = 0;
		// fourth bit
		t0 = GET_VALUE(6); t1 = GET_VALUE(7);
		temp_val = t0 < t1;
		val |= temp_val << 3;
		mask1_1 = GET_VALUE_MASK1(6); mask1_2 = GET_VALUE_MASK1(7); // mask1
		mask2_1 = GET_VALUE_MASK2(6); mask2_2 = GET_VALUE_MASK2(7); // mask2
		stable_val += (mask1_1 < mask1_2) ^ temp_val;
		stable_val += (mask2_1 < mask2_2) ^ temp_val;
		maskVal |= (stable_val == 0) << 3;
		stable_val = 0;
		// fifth bit
		t0 = GET_VALUE(8); t1 = GET_VALUE(9);
		temp_val = t0 < t1;
		val |= temp_val << 4;
		mask1_1 = GET_VALUE_MASK1(8); mask1_2 = GET_VALUE_MASK1(9); // mask1
		mask2_1 = GET_VALUE_MASK2(8); mask2_2 = GET_VALUE_MASK2(9); // mask2
		stable_val += (mask1_1 < mask1_2) ^ temp_val;
		stable_val += (mask2_1 < mask2_2) ^ temp_val;
		maskVal |= (stable_val == 0) << 4;
		stable_val = 0;
		// sixth bit
		t0 = GET_VALUE(10); t1 = GET_VALUE(11);
		temp_val = t0 < t1;
		val |= temp_val << 5;
		mask1_1 = GET_VALUE_MASK1(10); mask1_2 = GET_VALUE_MASK1(11); // mask1
		mask2_1 = GET_VALUE_MASK2(10); mask2_2 = GET_VALUE_MASK2(11); // mask2
		stable_val += (mask1_1 < mask1_2) ^ temp_val;
		stable_val += (mask2_1 < mask2_2) ^ temp_val;
		maskVal |= (stable_val == 0) << 5;
		stable_val = 0;
		// seventh bit
		t0 = GET_VALUE(12); t1 = GET_VALUE(13);
		temp_val = t0 < t1;
		val |= temp_val << 6;
		mask1_1 = GET_VALUE_MASK1(12); mask1_2 = GET_VALUE_MASK1(13); // mask1
		mask2_1 = GET_VALUE_MASK2(12); mask2_2 = GET_VALUE_MASK2(13); // mask2
		stable_val += (mask1_1 < mask1_2) ^ temp_val;
		stable_val += (mask2_1 < mask2_2) ^ temp_val;
		maskVal |= (stable_val == 0) << 6;
		stable_val = 0;
		// eigth bit
		t0 = GET_VALUE(14); t1 = GET_VALUE(15);
		temp_val = t0 < t1;
		val |= temp_val << 7;
		mask1_1 = GET_VALUE_MASK1(14); mask1_2 = GET_VALUE_MASK1(15); // mask1
		mask2_1 = GET_VALUE_MASK2(14); mask2_2 = GET_VALUE_MASK2(15); // mask2
		stable_val += (mask1_1 < mask1_2) ^ temp_val;
		stable_val += (mask2_1 < mask2_2) ^ temp_val;
		maskVal |= (stable_val == 0) << 7;

		descriptor[i] = (uchar)val;
		descMask[i] = (uchar)maskVal;
	}
#undef GET_VALUE
#undef GET_VALUE_MASK1
#undef GET_VALUE_MASK2

}


static void computeOrientation(const Mat& image,
	vector<KeyPoint>& keypoints,
	const vector<int>& umax)
{
	for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
		keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
	{
		keypoint->angle = IC_Angle(image, keypoint->pt, umax);
	}
}


void ExtractorNode_mdbrief::DivideNode(
	ExtractorNode_mdbrief &n1, 
	ExtractorNode_mdbrief &n2,
	ExtractorNode_mdbrief &n3,
	ExtractorNode_mdbrief &n4)
{
	const int halfX = ceil(static_cast<double>(UR.x - UL.x) / 2.0);
	const int halfY = ceil(static_cast<double>(BR.y - UL.y) / 2.0);

	//Define boundaries of childs
	n1.UL = UL;
	n1.UR = cv::Point2i(UL.x + halfX, UL.y);
	n1.BL = cv::Point2i(UL.x, UL.y + halfY);
	n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
	n1.vKeys.reserve(vKeys.size());

	n2.UL = n1.UR;
	n2.UR = UR;
	n2.BL = n1.BR;
	n2.BR = cv::Point2i(UR.x, UL.y + halfY);
	n2.vKeys.reserve(vKeys.size());

	n3.UL = n1.BL;
	n3.UR = n1.BR;
	n3.BL = BL;
	n3.BR = cv::Point2i(n1.BR.x, BL.y);
	n3.vKeys.reserve(vKeys.size());

	n4.UL = n3.UR;
	n4.UR = n2.BR;
	n4.BL = n3.BR;
	n4.BR = BR;
	n4.vKeys.reserve(vKeys.size());

	//Associate points to childs
	for (size_t i = 0; i<vKeys.size(); i++)
	{
		const cv::KeyPoint &kp = vKeys[i];
		if (kp.pt.x<n1.UR.x)
		{
			if (kp.pt.y<n1.BR.y)
				n1.vKeys.push_back(kp);
			else
				n3.vKeys.push_back(kp);
		}
		else if (kp.pt.y<n1.BR.y)
			n2.vKeys.push_back(kp);
		else
			n4.vKeys.push_back(kp);
	}

	if (n1.vKeys.size() == 1)
		n1.bNoMore = true;
	if (n2.vKeys.size() == 1)
		n2.bNoMore = true;
	if (n3.vKeys.size() == 1)
		n3.bNoMore = true;
	if (n4.vKeys.size() == 1)
		n4.bNoMore = true;

}

vector<cv::KeyPoint> mdBRIEFextractorOct::DistributeOctTree(
	const vector<cv::KeyPoint>& vToDistributeKeys,
	const int &minX,
	const int &maxX,
	const int &minY,
	const int &maxY,
	const int &N,
	const int &level)
{
	// Compute how many initial nodes   
	const int nIni = cvRound(static_cast<double>(maxX - minX) / (maxY - minY));

	const double hX = static_cast<double>(maxX - minX) / nIni;

	list<ExtractorNode_mdbrief> lNodes;

	vector<ExtractorNode_mdbrief*> vpIniNodes;
	vpIniNodes.resize(nIni);

	for (int i = 0; i<nIni; i++)
	{
		ExtractorNode_mdbrief ni;
		ni.UL = cv::Point2i(hX*static_cast<double>(i), 0);
		ni.UR = cv::Point2i(hX*static_cast<double>(i + 1), 0);
		ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
		ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
		ni.vKeys.reserve(vToDistributeKeys.size());

		lNodes.push_back(ni);
		vpIniNodes[i] = &lNodes.back();
	}

	//Associate points to childs
	for (size_t i = 0; i<vToDistributeKeys.size(); i++)
	{
		const cv::KeyPoint &kp = vToDistributeKeys[i];
		vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
	}

	list<ExtractorNode_mdbrief>::iterator lit = lNodes.begin();

	while (lit != lNodes.end())
	{
		if (lit->vKeys.size() == 1)
		{
			lit->bNoMore = true;
			lit++;
		}
		else if (lit->vKeys.empty())
			lit = lNodes.erase(lit);
		else
			lit++;
	}

	bool bFinish = false;

	int iteration = 0;

	vector<pair<int, ExtractorNode_mdbrief*> > vSizeAndPointerToNode;
	vSizeAndPointerToNode.reserve(lNodes.size() * 4);

	while (!bFinish)
	{
		iteration++;

		int prevSize = lNodes.size();

		lit = lNodes.begin();

		int nToExpand = 0;

		vSizeAndPointerToNode.clear();

		while (lit != lNodes.end())
		{
			if (lit->bNoMore)
			{
				// If node only contains one point do not subdivide and continue
				lit++;
				continue;
			}
			else
			{
				// If more than one point, subdivide
				ExtractorNode_mdbrief n1, n2, n3, n4;
				lit->DivideNode(n1, n2, n3, n4);

				// Add childs if they contain points
				if (n1.vKeys.size()>0)
				{
					lNodes.push_front(n1);
					if (n1.vKeys.size()>1)
					{
						nToExpand++;
						vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
						lNodes.front().lit = lNodes.begin();
					}
				}
				if (n2.vKeys.size()>0)
				{
					lNodes.push_front(n2);
					if (n2.vKeys.size()>1)
					{
						nToExpand++;
						vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
						lNodes.front().lit = lNodes.begin();
					}
				}
				if (n3.vKeys.size()>0)
				{
					lNodes.push_front(n3);
					if (n3.vKeys.size()>1)
					{
						nToExpand++;
						vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
						lNodes.front().lit = lNodes.begin();
					}
				}
				if (n4.vKeys.size()>0)
				{
					lNodes.push_front(n4);
					if (n4.vKeys.size()>1)
					{
						nToExpand++;
						vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
						lNodes.front().lit = lNodes.begin();
					}
				}

				lit = lNodes.erase(lit);
				continue;
			}
		}

		// Finish if there are more nodes than required features
		// or all nodes contain just one point
		if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
		{
			bFinish = true;
		}
		else if (((int)lNodes.size() + nToExpand * 3)>N)
		{

			while (!bFinish)
			{

				prevSize = lNodes.size();

				vector<pair<int, ExtractorNode_mdbrief*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
				vSizeAndPointerToNode.clear();

				sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
				for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
				{
					ExtractorNode_mdbrief n1, n2, n3, n4;
					vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

					// Add childs if they contain points
					if (n1.vKeys.size()>0)
					{
						lNodes.push_front(n1);
						if (n1.vKeys.size()>1)
						{
							vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
							lNodes.front().lit = lNodes.begin();
						}
					}
					if (n2.vKeys.size()>0)
					{
						lNodes.push_front(n2);
						if (n2.vKeys.size()>1)
						{
							vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
							lNodes.front().lit = lNodes.begin();
						}
					}
					if (n3.vKeys.size()>0)
					{
						lNodes.push_front(n3);
						if (n3.vKeys.size()>1)
						{
							vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
							lNodes.front().lit = lNodes.begin();
						}
					}
					if (n4.vKeys.size()>0)
					{
						lNodes.push_front(n4);
						if (n4.vKeys.size()>1)
						{
							vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
							lNodes.front().lit = lNodes.begin();
						}
					}

					lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

					if ((int)lNodes.size() >= N)
						break;
				}

				if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
					bFinish = true;

			}
		}
	}

	// Retain the best point in each node
	vector<cv::KeyPoint> vResultKeys;
	vResultKeys.reserve(nfeatures);
	for (list<ExtractorNode_mdbrief>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
	{
		vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
		cv::KeyPoint* pKP = &vNodeKeys[0];
		float maxResponse = pKP->response;

		for (size_t k = 1; k<vNodeKeys.size(); k++)
		{
			if (vNodeKeys[k].response>maxResponse)
			{
				pKP = &vNodeKeys[k];
				maxResponse = vNodeKeys[k].response;
			}
		}

		vResultKeys.push_back(*pKP);
	}

	return vResultKeys;
}

void mdBRIEFextractorOct::ComputeKeyPointsOctTree(
	vector<vector<KeyPoint> > *allKeypoints,
	vector<vector<KeyPoint*> > &keypoints_mul_GPU,
	cv::Mat *Mask)
{
	allKeypoints[0].resize(numlevels);
	allKeypoints[1].resize(numlevels);
	allKeypoints[2].resize(numlevels);
	
	gpuFast.detectAsync_mul(mvMultiImagePyramid[0].rowRange(minBordersY[0], maxBordersY[0]).colRange(minBordersX[0], maxBordersX[0]),
				mvMultiImagePyramid[0].rowRange(minBordersY[0]+mvMultiImagePyramid[0].rows/3, maxBordersY[0]+mvMultiImagePyramid[0].rows/3).colRange(minBordersX[0], maxBordersX[0]),
				mvMultiImagePyramid[0].rowRange(minBordersY[0]+mvMultiImagePyramid[0].rows/3*2, maxBordersY[0]+mvMultiImagePyramid[0].rows/3*2).colRange(minBordersX[0], maxBordersX[0]),
				0);
	
	for (int level = 0; level < numlevels; ++level)
	{
		vector<cv::KeyPoint> vToDistributeKeys[CAMS];
		vToDistributeKeys[0].reserve(nfeatures * 10);
		vToDistributeKeys[1].reserve(nfeatures * 10);
		vToDistributeKeys[2].reserve(nfeatures * 10);
		
		if (!Mask[0].empty())
		{
			gpuFast.joinDetectAsync_mul(vToDistributeKeys,
						    mvMaskPyramid[level*CAMS+0].rowRange(minBordersY[level], maxBordersY[level]).colRange(minBordersX[level], maxBordersX[level]),
						    mvMaskPyramid[level*CAMS+1].rowRange(minBordersY[level], maxBordersY[level]).colRange(minBordersX[level], maxBordersX[level]),
						    mvMaskPyramid[level*CAMS+2].rowRange(minBordersY[level], maxBordersY[level]).colRange(minBordersX[level], maxBordersX[level]));
		}
		else
		{
			gpuFast.joinDetectAsync(vToDistributeKeys[0]);
			gpuFast.joinDetectAsync(vToDistributeKeys[1]);
			gpuFast.joinDetectAsync(vToDistributeKeys[2]);
		}
		
		if(level==0)
			mcvStream[0].waitForCompletion();

		if (level+1 < numlevels)
		{
			gpuFast.detectAsync_mul(mvMultiImagePyramid[level+1].rowRange(minBordersY[level + 1], maxBordersY[level + 1]).colRange(minBordersX[level + 1], maxBordersX[level + 1]),
						mvMultiImagePyramid[level+1].rowRange(minBordersY[level + 1]+mvMultiImagePyramid[level+1].rows/3, maxBordersY[level + 1]+mvMultiImagePyramid[level+1].rows/3).colRange(minBordersX[level + 1], maxBordersX[level + 1]),
						mvMultiImagePyramid[level+1].rowRange(minBordersY[level + 1]+mvMultiImagePyramid[level+1].rows/3*2, maxBordersY[level + 1]+mvMultiImagePyramid[level+1].rows/3*2).colRange(minBordersX[level + 1], maxBordersX[level + 1]),
						level+1);
		}
		
		if (level != 0)
		{
			ic_angle.launch_async_mul(mvMultiImagePyramid, allKeypoints,keypoints_mul_GPU, HALF_PATCH_SIZE, minBordersX[level-1], minBordersY[level-1], level-1, PATCH_SIZE * mvScaleFactor[level-1]);
			
			for(int c=0;c<3;c++)
			{
				mpGaussianFilter->apply(mvMultiImagePyramid[level-1].rowRange(mvMultiImagePyramid[level-1].rows/3*c, mvMultiImagePyramid[level-1].rows/3*(c+1)), mvMultiImagePyramid[level-1].rowRange(mvMultiImagePyramid[level-1].rows/3*c, mvMultiImagePyramid[level-1].rows/3*(c+1)), ic_angle.cvStream(0));
			}
		}

		for(int c=0;c<3;c++)
		{
			vector<KeyPoint> & keypoints = allKeypoints[c][level];
			keypoints.reserve(nfeatures);

			PUSH_RANGE("DistributeOctTree", 3);
			keypoints = DistributeOctTree(vToDistributeKeys[c], minBordersX[level], maxBordersX[level], minBordersY[level], maxBordersY[level],mnFeaturesPerLevel[level], level);
			POP_RANGE;
		}
	}
	
	ic_angle.launch_async_mul(mvMultiImagePyramid, allKeypoints,keypoints_mul_GPU, HALF_PATCH_SIZE, minBordersX[numlevels-1], minBordersY[numlevels-1], numlevels-1, PATCH_SIZE * mvScaleFactor[numlevels-1]);
	
	for(int c=0;c<3;c++)
	{
		mpGaussianFilter->apply(mvMultiImagePyramid[numlevels-1].rowRange(mvMultiImagePyramid[numlevels-1].rows/3*c, mvMultiImagePyramid[numlevels-1].rows/3*(c+1)), mvMultiImagePyramid[numlevels-1].rowRange(mvMultiImagePyramid[numlevels-1].rows/3*c, mvMultiImagePyramid[numlevels-1].rows/3*(c+1)), ic_angle.cvStream(0));
	}
}

void mdBRIEFextractorOct::ComputePyramid(
	vector<Mat> image, 
	Mat *Mask)
{
	if (mvImagePyramidAllocatedFlag == false)
	{
		int minX[CAMS][numlevels],maxX[CAMS][numlevels];//,minY[CAMS],maxY[CAMS];
		for (int level = 0; level < numlevels; ++level)
		{
			float scale = mvInvScaleFactor[level];
			Size sz(cvRound((float)image[0].cols*scale), cvRound((float)image[0].rows*scale));
			Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
			
			Size multi_sz(cvRound((float)image[0].cols*scale), cvRound((float)image[0].rows*scale*3));
			Size multi_wholeSize(multi_sz.width + EDGE_THRESHOLD*2, multi_sz.height + EDGE_THRESHOLD*2);
			cuda::GpuMat multi_target(multi_wholeSize, image[0].type(), ORB_SLAM2::cuda::gpu_mat_allocator);
			mvMultiImagePyramidBorder[level]=multi_target;
			mvMultiImagePyramid[level]=multi_target(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, multi_sz.width, multi_sz.height));
			
			// first frame, allocate the Pyramids
			for(int c=0;c<CAMS;c++)
			{
				if (!Mask[c].empty())
				{
					Mat masktemp(wholeSize, Mask[c].type());
					mvMaskPyramid[level*CAMS+c]=masktemp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
				
					if (level != 0)
					{
						resize(mvMaskPyramid[(level-1)*CAMS+c], mvMaskPyramid[level*CAMS+c], sz, 0, 0, INTER_NEAREST);
						copyMakeBorder(mvMaskPyramid[level*CAMS+c], masktemp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
							       EDGE_THRESHOLD,BORDER_CONSTANT + BORDER_ISOLATED);
					}
					else
					{
						copyMakeBorder(Mask[c], masktemp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
							       EDGE_THRESHOLD,BORDER_CONSTANT + BORDER_ISOLATED);
					}
				
					int set;
					set=1;
					for(minX[c][level]=0;set&&minX[c][level]<mvMaskPyramid[level*CAMS+c].cols;minX[c][level]++)
						for(int t=0;set&&t<mvMaskPyramid[level*CAMS+c].rows;t++)
							if(mvMaskPyramid[level*CAMS+c].at<uchar>(t,minX[c][level])==255)
							{
								set=0;
								minX[c][level]-=3;
								minX[c][level]--;
							}
				
					set=1;
					for(maxX[c][level]=mvMaskPyramid[level*CAMS+c].cols-1;set&&maxX[c][level]>=0;maxX[c][level]--)
						for(int t=0;set&&t<mvMaskPyramid[level*CAMS+c].rows;t++)
							if(mvMaskPyramid[level*CAMS+c].at<uchar>(t,maxX[c][level])==255)
							{
								set=0;
								maxX[c][level]+=4;
								maxX[c][level]++;
							}
				}
			}
		}
		
		for(int level=0;level<numlevels;level++)
		{
			int minBorderX = EDGE_THRESHOLD - 3;
			int minBorderY = minBorderX;
			int maxBorderX = mvMultiImagePyramid[level].cols - EDGE_THRESHOLD + 3;
			int maxBorderY = mvMultiImagePyramid[level].rows/3 - EDGE_THRESHOLD + 3;
		
			int minMaskBorderX=mvMultiImagePyramid[level].cols;
			int maxMaskBorderX=0;
	
			for(int c=0;c<3;c++)
			{
				if(minMaskBorderX>minX[c][level])minMaskBorderX=minX[c][level];
				if(maxMaskBorderX<maxX[c][level])maxMaskBorderX=maxX[c][level];
			}
		
			minBordersX[level]=minMaskBorderX>minBorderX?minMaskBorderX:minBorderX;
			maxBordersX[level]=maxMaskBorderX<maxBorderX?maxMaskBorderX:maxBorderX;
			minBordersY[level]=minBorderY;
			maxBordersY[level]=maxBorderY;
		}
		
		mpGaussianFilter = cv::cuda::createGaussianFilter(mvMultiImagePyramid[0].type(), mvMultiImagePyramid[0].type(), Size(7, 7), 2, 2, BORDER_REFLECT_101);
		mvImagePyramidAllocatedFlag = true;
	}
	
	Mat multiImage;
	multiImage.push_back(image[0]);
	multiImage.push_back(image[1]);
	multiImage.push_back(image[2]);
	
	for (int level = 0; level < numlevels; ++level)
	{
		float scale = mvInvScaleFactor[level];
		Size sz(cvRound((float)multiImage.cols*scale), cvRound((float)multiImage.rows*scale));
		cuda::GpuMat target(mvMultiImagePyramidBorder[level]);

		if (level != 0)
		{
			cuda::resize(mvMultiImagePyramid[level-1], mvMultiImagePyramid[level], sz, 0, 0, INTER_LINEAR, mcvStream[0]);

			cuda::copyMakeBorder(mvMultiImagePyramid[level], mvMultiImagePyramidBorder[level], EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
					     EDGE_THRESHOLD,BORDER_REFLECT_101, cv::Scalar(), mcvStream[0]);
		}
		else
		{
			cuda::GpuMat gpuImg(multiImage);

			cuda::copyMakeBorder(gpuImg, mvMultiImagePyramidBorder[level], EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
					     EDGE_THRESHOLD,BORDER_REFLECT_101, cv::Scalar(), mcvStream[0]);
			mcvStream[0].waitForCompletion();
		}
	}
}

void mdBRIEFextractorOct::operator()(
	const vector<Mat>& _image,
	vector<cCamModelGeneral_>& _camModel,
	vector<vector<KeyPoint> >& _keypoints,
	vector<Mat>& _descriptors,
	vector<Mat>& _descriptorMasks
	)
{
	PUSH_RANGE("mdBRIEFextractorOct", 0);
	
	if (_image[0].empty()||_image[1].empty()||_image[2].empty())
		return;

	Mat mask[CAMS];
	mask[0]= _camModel[0].GetMirrorMask(0);
	mask[1]= _camModel[1].GetMirrorMask(0);
	mask[2]= _camModel[2].GetMirrorMask(0);
	
	assert(_image[0].type() == CV_8UC1);
	assert(_image[1].type() == CV_8UC1);
	assert(_image[2].type() == CV_8UC1);

	ComputePyramid(_image, mask);
	
	vector<vector<KeyPoint> > allKeypoints[CAMS];
	vector<vector<KeyPoint*> > keypoints_mul_GPU(8,vector<KeyPoint*>(3));
	ComputeKeyPointsOctTree(allKeypoints,keypoints_mul_GPU,mask);
	
	for(int cc=0;cc<CAMS;cc++)
	{
		int nkeypoints = 0;
		for (int level = 0; level < numlevels; ++level)
		{
			nkeypoints += (int)allKeypoints[cc][level].size();
		}
		
		if (nkeypoints == 0)
		{
			_descriptors[cc].release();
			_descriptorMasks[cc].release();
		}
		else
		{
			_descriptors[cc].create(nkeypoints, descSize, CV_8U);
			_descriptorMasks[cc].create(nkeypoints, descSize, CV_8U);
		}

		_keypoints[cc].clear();
		_keypoints[cc].reserve(nkeypoints);
	}
	
	int offset[CAMS] = {0,0,0};
	for(int cc=0;cc<CAMS;cc++)
	{
		vector<KeyPoint>& keypoints = allKeypoints[cc][0];
		gpuOrb.launch_async(mvMultiImagePyramid[0], keypoints.data(), keypoints.size(),keypoints_mul_GPU[0],mvScaleFactor[0],cc);
	}
	
	for (int level = 0; level < numlevels; ++level)
	{
		for(int cc=0;cc<CAMS;cc++)
		{
			Mat desc = _descriptors[cc].rowRange(offset[cc], offset[cc] + allKeypoints[cc][level].size());
			gpuOrb.join(desc,allKeypoints[cc][level],keypoints_mul_GPU[level],cc);
			offset[cc] += allKeypoints[cc][level].size();
		}
		
		if (level + 1 < numlevels)
		{
			for(int cc=0;cc<CAMS;cc++)
			{
				vector<KeyPoint>& keypoints = allKeypoints[cc][level+1];
				gpuOrb.launch_async(mvMultiImagePyramid[level+1], keypoints.data(), keypoints.size(),keypoints_mul_GPU[level+1],mvScaleFactor[level+1],cc);
			}
		}
		
		for(int cc=0;cc<CAMS;cc++)
		// And add the keypoints to the output
		_keypoints[cc].insert(_keypoints[cc].end(), allKeypoints[cc][level].begin(), allKeypoints[cc][level].end());
	}
	
	POP_RANGE;
}
}
