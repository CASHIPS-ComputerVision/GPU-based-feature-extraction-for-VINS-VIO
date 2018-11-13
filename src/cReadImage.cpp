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

#include "cReadImage.h"

#include <opencv2/highgui/highgui.hpp>

#include <iostream>

namespace MultiColSLAM
{
	using namespace std;
	
	cReadImage::cReadImage(vector<vector<string>> vImgFilenames,int bufferSize):
		mvImgFilenames(vImgFilenames),
		bufferSizeAct(bufferSize),
		sizeCount(0),
		putCount(0),
		getCount(0),
		mbFinishRequested(false),
		mbFinished(false)
	{
		nImages = mvImgFilenames[0].size();
		nrCams = static_cast<int>(mvImgFilenames.size());
		
		for(int i=0;i<bufferSizeAct;i++)
		{
			mvImgs[i].resize(nrCams);
			imgReady[i]=false;
		}
	}

	void cReadImage::Run()
	{
		for (int ni = 0; ni < nImages; ni++)
		{
			for (int c = 0; c < nrCams; ++c)
			{
				mvImgs[putCount][c] = cv::imread(mvImgFilenames[c][ni], CV_LOAD_IMAGE_GRAYSCALE);
				if (mvImgs[putCount][c].empty())
				{
					cerr << endl << "Failed to load image at: " << mvImgFilenames[c][ni] << endl;
					break;
				}
			}
			
			setReady();

			while(checkReady())
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}putCount=(putCount+1)%bufferSizeAct;
			
			if (CheckFinish())
				break;
		}
		SetFinish();
	}
	
	void cReadImage::setReady()
	{
		std::unique_lock<std::mutex> lock(mMutexReady);
		imgReady[putCount]=true;
	}
	
	bool cReadImage::checkReady()
	{
		std::unique_lock<std::mutex> lock(mMutexReady);
		std::unique_lock<std::mutex> lock1(mMutexFinish);
		return imgReady[(putCount+1)%bufferSizeAct]&(!mbFinishRequested);
	}
	
	bool cReadImage::isReady(vector<cv::Mat> &vImgs)
	{
		std::unique_lock<std::mutex> lock(mMutexReady);
		if(imgReady[getCount])
		{
			vImgs.assign(mvImgs[getCount].begin(),mvImgs[getCount].end());
			imgReady[getCount]=false;
			
			getCount=(getCount+1)%bufferSizeAct;
			
			return true;
		}
		return false;
	}

	void cReadImage::RequestFinish()
	{
		std::unique_lock<std::mutex> lock(mMutexFinish);
		mbFinishRequested = true;
	}

	bool cReadImage::CheckFinish()
	{
		std::unique_lock<std::mutex>  lock(mMutexFinish);
		return mbFinishRequested;
	}

	void cReadImage::SetFinish()
	{
		std::unique_lock<std::mutex>  lock(mMutexFinish);
		mbFinished = true;
	}

	bool cReadImage::isFinished()
	{
		std::unique_lock<std::mutex>  lock(mMutexFinish);
		return mbFinished;
	}


}


