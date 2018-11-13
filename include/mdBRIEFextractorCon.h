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

#ifndef MDBRIEFEXTRACTORCON_H
#define MDBRIEFEXTRACTORCON_H

#include <mutex>
#include <thread>
#include <opencv2/core/core.hpp>

#include "cReadImage.h"
#include "mdBRIEFextractorOct.h"
#include "cam_system_omni.h"

namespace MultiColSLAM
{
	using namespace std;
	
	class cReadImage;
	
	class mdBRIEFextractorCon
	{
	public:
	  
		mdBRIEFextractorCon(cReadImage* pReadImage,vector<vector<string>> vImgFilenames,cMultiCamSys_ camSystem_,string sSettingsPath,int bufferSize);
		
		void Run();
		
		bool isReady(vector<cv::Mat> &vImgs,
			     vector<vector<cv::KeyPoint>> &keyPtsTemp,
			     vector<cv::Mat> &Descriptors,
			     vector<cv::Mat> &DescriptorMasks,
			     mdBRIEFextractorOct* p_mdBRIEF_extractorOct);
		
		void setExtractorOct(mdBRIEFextractorOct* p_mdBRIEF_extractorOct);

		void RequestFinish();

		bool isFinished();
		
	protected:
	  
		const static int bufferSizeCon=10;
		int bufferSizeAct;

		bool CheckFinish();
		void SetFinish();

		bool mbFinishRequested;
		bool mbFinished;
		std::mutex mMutexFinish;
		
		cReadImage* mpReadImage;
		
		vector<vector<string>> mvImgFilenames;
		int nImages;
		int nrCams;
		
		vector<cv::Mat> mvImgs[bufferSizeCon];
		
		vector<vector<cv::KeyPoint>> mkeyPtsTemp[bufferSizeCon];
		
		// descriptors for each camera
		vector<cv::Mat> mDescriptors[bufferSizeCon];
		// learned descriptor masks
		vector<cv::Mat> mDescriptorMasks[bufferSizeCon];
		
		vector<cCamModelGeneral_> camModel;
		
		// camera system class
		cMultiCamSys_ camSystem;
		int numberCameras;
		
		std::string msSettingsPath;
		// features
		bool use_mdBRIEF;
		// mdBRIEF with octree
		mdBRIEFextractorOct* mp_mdBRIEF_extractorOct;
		
		int putCount;
		int getCount;
		
		void setReady();
		bool checkReady();
		
		bool imgReady[bufferSizeCon];
		std::mutex mMutexReady;
		
		int img_row;
		int img_col;
		unsigned char *img1_gpu;
	};

}
#endif // MDBRIEFEXTRACTORCON_H
