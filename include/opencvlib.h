#if !defined _HEADER_OPENCV_LIB_20130404_INCLUDED_
#define _HEADER_OPENCV_LIB_20130404_INCLUDED_

#include "opencv2/opencv.hpp"
using namespace cv;

#define OPENCV_VERSION   CVAUX_STR(CV_VERSION_EPOCH) CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR)

#ifdef _DEBUG
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_core"OPENCV_VERSION"d.lib")
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_imgproc"OPENCV_VERSION"d.lib")
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_highgui"OPENCV_VERSION"d.lib")
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_legacy"OPENCV_VERSION"d.lib")
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_objdetect"OPENCV_VERSION"d.lib")
#else
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_core"OPENCV_VERSION".lib")
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_imgproc"OPENCV_VERSION".lib")
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_highgui"OPENCV_VERSION".lib")
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_legacy"OPENCV_VERSION".lib")
	#pragma comment(lib,"..\\..\\3rdparty\\lib\\opencv_objdetect"OPENCV_VERSION".lib")
#endif

#endif