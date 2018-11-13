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

#include "mdBRIEFextractorCon.h"

#include <opencv2/highgui/highgui.hpp>

#include <iostream>

void startgpu(cv::Mat image,unsigned char *img1_gpu);
void endgpu(unsigned char *img1_gpu);
void addgpu(int img_row,int img_col,unsigned char *img1_gpu);

namespace MultiColSLAM
{
	using namespace std;
	
	mdBRIEFextractorCon::mdBRIEFextractorCon(cReadImage* pReadImage,vector<vector<string>> vImgFilenames,cMultiCamSys_ camSystem_,string sSettingsPath,int bufferSize):
		mpReadImage(pReadImage),
		mvImgFilenames(vImgFilenames),
		camSystem(camSystem_),
		msSettingsPath(sSettingsPath),
		bufferSizeAct(bufferSize),
		putCount(0),
		getCount(0),
		mbFinishRequested(false),
		mbFinished(false)
	{
		nImages = mvImgFilenames[0].size();
		nrCams = static_cast<int>(mvImgFilenames.size());
		
		cv::Mat image=cv::imread(mvImgFilenames[0][0], CV_LOAD_IMAGE_GRAYSCALE);
		if (image.empty())
		{
			cerr << endl << "Failed to load image at: " << mvImgFilenames[0][0] << endl;
		}
		img_row=image.rows;
		img_col=image.cols;
		startgpu(image,img1_gpu);
		
		for(int i=0;i<bufferSizeAct;i++)
		{
			mvImgs[i].resize(nrCams);
			mkeyPtsTemp[i].resize(nrCams);
			mDescriptors[i].resize(nrCams);
			mDescriptorMasks[i].resize(nrCams);
			
			imgReady[i]=false;
		}
		
		camModel.resize(nrCams);
		camModel[0] = camSystem.GetCamModelObj(0);
		camModel[1] = camSystem.GetCamModelObj(1);
		camModel[2] = camSystem.GetCamModelObj(2);
		
		numberCameras = static_cast<int>(camSystem.Get_All_M_c().size());
		
		cv::FileStorage slamSettings(msSettingsPath, cv::FileStorage::READ);
		// Load ORB parameters
		int featDim = (int)slamSettings["extractor.descSize"];
		int nFeatures = (int)slamSettings["extractor.nFeatures"];
		float fScaleFactor = slamSettings["extractor.scaleFactor"];
		int nLevels = (int)slamSettings["extractor.nLevels"];
		int fastTh = (int)slamSettings["extractor.fastTh"];
		int Score = (int)slamSettings["extractor.nScoreType"];
		
		assert(Score == 1 || Score == 0);
		
		this->use_mdBRIEF = false;
		bool learnMasks = false;
		
		int usemd = (int)slamSettings["extractor.usemdBRIEF"];
		this->use_mdBRIEF = static_cast<bool>(usemd);
		int masksL = (int)slamSettings["extractor.masks"];
		learnMasks = static_cast<bool>(masksL);
		
		int useAgast = (int)slamSettings["extractor.useAgast"];
		int fastAgastType = (int)slamSettings["extractor.fastAgastType"];
		int descSize = (int)slamSettings["extractor.descSize"];
	}
	
	void mdBRIEFextractorCon::setExtractorOct(mdBRIEFextractorOct* p_mdBRIEF_extractorOct)
	{
		mp_mdBRIEF_extractorOct=p_mdBRIEF_extractorOct;
	}

	void mdBRIEFextractorCon::Run()
	{
		while (1)
		{
			addgpu(img_row,img_col,img1_gpu);
			
			bool readFinsh=false;
			while(!mpReadImage->isReady(mvImgs[putCount]))
			{
				if(mpReadImage->isFinished())
				{
					readFinsh=true;
					break;
				}
				addgpu(img_row,img_col,img1_gpu);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			if (readFinsh)
				break;
			
			(*mp_mdBRIEF_extractorOct)(mvImgs[putCount], camModel,mkeyPtsTemp[putCount],
						   mDescriptors[putCount], mDescriptorMasks[putCount]);
			
			setReady();
			
			putCount=(putCount+1)%bufferSizeAct;

			while(checkReady())
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			
			if (CheckFinish())
				break;
		}
		endgpu(img1_gpu);
		SetFinish();
	}
	
	void mdBRIEFextractorCon::setReady()
	{
		std::unique_lock<std::mutex> lock(mMutexReady);
		
		imgReady[putCount]=true;
	}
	
	bool mdBRIEFextractorCon::checkReady()
	{
		std::unique_lock<std::mutex> lock(mMutexReady);
		std::unique_lock<std::mutex> lock1(mMutexFinish);
		return imgReady[putCount]&&(!mbFinishRequested);
	}
	
	bool mdBRIEFextractorCon::isReady(vector<cv::Mat> &vImgs,
					  vector<vector<cv::KeyPoint>> &keyPtsTemp,
					  vector<cv::Mat> &Descriptors,
					  vector<cv::Mat> &DescriptorMasks,
					  mdBRIEFextractorOct* p_mdBRIEF_extractorOct)
	{
		std::unique_lock<std::mutex> lock(mMutexReady);
		
		if(imgReady[getCount])
		{
			mp_mdBRIEF_extractorOct=p_mdBRIEF_extractorOct;

			vImgs.assign(mvImgs[getCount].begin(),mvImgs[getCount].end());
			keyPtsTemp.assign(mkeyPtsTemp[getCount].begin(),mkeyPtsTemp[getCount].end());
			Descriptors.assign(mDescriptors[getCount].begin(),mDescriptors[getCount].end());
			DescriptorMasks.assign(mDescriptorMasks[getCount].begin(),mDescriptorMasks[getCount].end());
			
			imgReady[getCount]=false;
			getCount=(getCount+1)%bufferSizeAct;
			return true;
		}
		return false;
	}

	void mdBRIEFextractorCon::RequestFinish()
	{
		std::unique_lock<std::mutex> lock(mMutexFinish);
		mbFinishRequested = true;
	}

	bool mdBRIEFextractorCon::CheckFinish()
	{
		std::unique_lock<std::mutex>  lock(mMutexFinish);
		return mbFinishRequested;
	}

	void mdBRIEFextractorCon::SetFinish()
	{
		std::unique_lock<std::mutex>  lock(mMutexFinish);
		mbFinished = true;
	}

	bool mdBRIEFextractorCon::isFinished()
	{
		std::unique_lock<std::mutex>  lock(mMutexFinish);
		return mbFinished;
	}


}


