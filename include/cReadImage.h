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

#ifndef CREADIMAGE_H
#define CREADIMAGE_H

#include <mutex>
#include <thread>
#include <opencv2/core/core.hpp>

namespace MultiColSLAM
{
	using namespace std;
	
	class cReadImage
	{
	public:
	  
		cReadImage(vector<vector<string>> vImgFilenames,int bufferSize);
		
		void Run();
		
		bool isReady(vector<cv::Mat> &vImgs);

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
		
		vector<vector<string>> mvImgFilenames;
		int nImages;
		int nrCams;
		
		vector<cv::Mat> mvImgs[bufferSizeCon];
		
		int sizeCount;
		
		int putCount;
		int getCount;
		
		void setReady();
		bool checkReady();
		
		bool imgReady[bufferSizeCon];
		std::mutex mMutexReady;
	};

}
#endif // CREADIMAGE_H
