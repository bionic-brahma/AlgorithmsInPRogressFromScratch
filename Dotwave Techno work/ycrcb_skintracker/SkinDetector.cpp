#include "SkinDetector.h"
#include"opencv2\opencv.hpp"
using namespace cv;
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
SkinDetector::SkinDetector(void)
{
//YCrCb threshold
//You can change the values and see what happens
Y_MIN  = 0;
Y_MAX  = 255;
Cr_MIN = 137;
Cr_MAX = 177;
Cb_MIN = 77;
Cb_MAX = 127;
}

SkinDetector::~SkinDetector(void)
{
}

//this function will return a skin masked image
cv::Mat SkinDetector::getSkin(cv::Mat input)
{
cv::Mat skin;
//do some blurr and histoanalysis


//cv::GaussianBlur(input,input,cv::Size(3,3),1.1);

//first convert our RGB image to YCrCb
cv::cvtColor(input,skin,cv::COLOR_BGR2YCrCb);

//uncomment the following line to see the image in YCrCb Color Space
//cv::imshow("YCrCb Color Space",skin);

//filter the image in YCrCb color space
cv::inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);
skin.convertTo(skin,-1,1.5,-20);
cv::Mat element = cv::getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
int operation = morph_operator + 3;
cv::blur(skin,skin,cv::Size(5,5));
cv::dilate(skin,skin,element);
cv::erode(skin,skin,element);


return skin;
}