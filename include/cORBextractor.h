/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


//namespace ORB_SLAM
//{

class ExtractorNode
{
public:
	ExtractorNode() :bNoMore(false){}

	void DivideNode(ExtractorNode &n1, 
		ExtractorNode &n2, 
		ExtractorNode &n3, 
		ExtractorNode &n4);

	std::vector<cv::KeyPoint> vKeys;
	cv::Point2i UL, UR, BL, BR;
	std::list<ExtractorNode>::iterator lit;
	bool bNoMore;
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

	ORBextractor(int nfeatures = 1000, 
		double scaleFactor = 1.2, 
		int nlevels = 8, 
		int scoreType = FAST_SCORE, 
		int fastTh = 20);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels;}

    double inline GetScaleFactor(){
        return scaleFactor;}


protected:
	void ComputePyramid(cv::Mat image, cv::Mat Mask = cv::Mat());
	void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
	std::vector<cv::KeyPoint> DistributeOctTree(
		const std::vector<cv::KeyPoint>& vToDistributeKeys, 
		const int &minX,
		const int &maxX, 
		const int &minY, 
		const int &maxY, 
		const int &nFeatures, 
		const int &level);

	void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

    //void ComputePyramid(cv::Mat image, cv::Mat Mask=cv::Mat());
    //void ComputeKeyPoints(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int scoreType;
    int fastTh;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

	std::vector<double> mvScaleFactor;
	std::vector<double> mvInvScaleFactor;

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<cv::Mat> mvMaskPyramid;

};

//} //namespace ORB_SLAM

#endif

