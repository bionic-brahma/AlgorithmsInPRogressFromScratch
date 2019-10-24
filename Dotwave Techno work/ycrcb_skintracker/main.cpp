#include<opencv2\opencv.hpp>
#include"SkinDetector.h"
#include<opencv\cv.h>
#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>
#include<dlib\image_processing\shape_predictor.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include<conio.h>
#include <dlib/opencv.h>

using namespace std;
using namespace cv;
using namespace dlib;

shape_predictor testpredictor;
CascadeClassifier facedetector;
CascadeClassifier eyecenter;

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

std::vector<Rect> detectfaces(Mat img){

    std::vector<Rect> faces;
    facedetector.detectMultiScale(img, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE,Size(2,2));

    return faces;
}

std::vector<Point> detecteyescenter(Mat img){

    std::vector<Rect> eyes;
	eyecenter.detectMultiScale(img, eyes, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE,Size(2,2));
	std::vector<cv::Point> center;
	for(int i=0;i<eyes.size();i++){
		center.push_back(cv::Point(eyes[i].width/2,eyes[i].height/2));
	}
	if(center.size()==0){
		center.push_back(Point(-1,-1));
		
	}
    return center;
}


//file reader
std::vector<string> files(string folder)
{
    std::vector<string> names;
	folder= "dir "+folder+ " /b > test.txt";
	system(folder.c_str());
	//parsing of test.txt///  ************* make sure that this file test.txt is not deleted till the process ********
   ifstream file;
   file.open("test.txt",std::ios::in);
   string out;
  // char* outchar="";
   while(std::getline(file,out)){
   
   //out= outchar;
   //test
   cout<<out<<"\n";
   names.push_back(out);
   }
   file.close();
   return names;

}
//
  // global variable for testing purpose
  // Please ignore in real application process
int face_problem_count=0;
int false_fw_detection=0;
int main()
{
	double meacj=0;
	double meafj=0;
	//namedWindow("dlib",1);
	dlib::deserialize("shape_predictor_68_face_landmarks.dat")>>testpredictor;
	string addr= "Heart\\";
	std::vector<string> file= files(addr);
	facedetector.load("haarcascade_frontalface_alt_tree.xml");
	//eyecenter.load("haarcascade_eye_tree_eyeglasses.xml");



Mat cameraFeed;
SkinDetector mySkinDetector;

Mat skinMat;
Mat img,processed;

int iter= file.size();
int k=0;
while(k<file.size()){
	try{
		string imgfile= addr+"\\"+ file[k];
		k++;
//store image to matrix
cameraFeed= imread(imgfile,1);
cv::resize(cameraFeed,cameraFeed,cv::Size(480,480));

cv::resize(cameraFeed,img,cv::Size(cameraFeed.cols/4,cameraFeed.rows/4));
cvtColor(img,img,CV_BGR2GRAY);
std::vector<cv::Rect> faces= detectfaces(img);
//std::vector<cv::Point> center= detecteyescenter(img);
if(faces.size()!=1){
	face_problem_count++;
	cout<<"\n[x] Align the camera in such a way that there is only one face in the frame.";
}
else{
	int factor =4;

//show the current image
	// there is an exception of memory for 
	 //processing the image for writting in the memory
           // faces[0].x=faces[0].x+0.05*faces[0].width;    //////////////////////
            //faces[0].width=faces[0].width-0.05*faces[0].width;
           // faces[0].y=faces[0].y-0.04*faces[0].height;
			//faces[0].height= faces[0].height+0.027*faces[0].width;
Rect oriface= Rect(Point(factor*faces[0].x,factor*faces[0].y),Point(factor*faces[0].x+factor*faces[0].width,factor*faces[0].y+factor*faces[0].height));
processed= cameraFeed(oriface);

Mat grayprocessed;


cv_image<bgr_pixel> cimg(processed);

full_object_detection shape= testpredictor(cimg,openCVRectToDlib(oriface));

int i=0;
if(shape.num_parts()!=68){
	cout<<"\n**************************************dlib model broked due to constrains******************************";
}
std::vector<cv::Point>dlib68_landmarkpoints;
			while(i<68){
				dlib68_landmarkpoints.push_back(Point(shape.part(i).x(),shape.part(i).y()));
				i++;
			}
			//cv::line(cameraFeed,dlib68_landmarkpoints[1],dlib68_landmarkpoints[15], cv::Scalar(0,0,255));
			//cv::line(cameraFeed,dlib68_landmarkpoints[4],dlib68_landmarkpoints[12], cv::Scalar(0,0,255));




skinMat= mySkinDetector.getSkin(processed);

std::vector<std::vector<cv::Point>> closed;
cv::findContours(skinMat,closed,CV_RETR_LIST ,CV_CHAIN_APPROX_SIMPLE);
double temp= 0; int maxloc=0;

for(int i=0;i<closed.size();i++){
	if(cv::contourArea(closed[i])>temp){
		temp= cv::contourArea(closed[i]);
		maxloc=i;
	}
}

int topx=1000; int topy=1000;
int lowx=0; int lowy=0;
int leftx=1000; int lefty=1000;
int rightx=0; int righty=0;
int height=0;
int width=0;

cv::approxPolyDP(closed[maxloc],closed[maxloc],0.4,false);

for(long long x=0;x<closed[maxloc].size();x++){
	if(lowy<closed[maxloc][x].y){
		lowy=closed[maxloc][x].y;
		lowx=closed[maxloc][x].x;
	}
	if(topy>closed[maxloc][x].y){
		topy=closed[maxloc][x].y;
		topx=closed[maxloc][x].x;
	}
	if(leftx>closed[maxloc][x].x){
		leftx=closed[maxloc][x].x;
		lefty=closed[maxloc][x].y;
	}
	if(rightx<closed[maxloc][x].x){
		rightx=closed[maxloc][x].x;   // max error chance is 10%
		righty=closed[maxloc][x].y;
	}
	//cv::putText(cameraFeed(oriface),std::to_string(x),closed[maxloc][x],1,1,Scalar(0,0,255));
}

 
int h=0;

h= (dlib68_landmarkpoints[8].y-topy);
system("CLS");



cv::approxPolyDP(closed[maxloc],closed[maxloc],0.4,true);

//cout<<"\nlandmark height mark: "<<dlib68_landmarkpoints[8].y;


waitKey(30);

int cw= dlib68_landmarkpoints[15].x-dlib68_landmarkpoints[1].x;
int jw= (int)(-1*(dlib68_landmarkpoints[4].x-dlib68_landmarkpoints[12].x+dlib68_landmarkpoints[5].x-dlib68_landmarkpoints[11].x)/2);
int fw=0;

int fh= max(std::abs(dlib68_landmarkpoints[19].y-topy),std::abs(dlib68_landmarkpoints[24].y-topy));							//optimization chances
int top_shift_for_fw= fh/2;
std::vector<cv::Point> fw_points;
int fwleft=1000;
int fwright=-1;
int fw_y= topy+top_shift_for_fw;
//cv::line(cameraFeed(oriface),cv::Point(topx,fw_y),cv::Point(topx,topy),cv::Scalar(0,20,255));


cout<<"\nforehead hight:  "<<fw_y;
for(long long x=0;x<closed[maxloc].size();x++){
   if(fw_y-5<=closed[maxloc][x].y<=fw_y+5){
	   if( fwleft>closed[maxloc][x].x)// problem at fw estimation date 22 dec 2016
	   fwleft=closed[maxloc][x].x;
	   if( fwright<closed[maxloc][x].x)
	   fwright=closed[maxloc][x].x;
	   fw_points.push_back(closed[maxloc][x]); // be used to find the groups of hairs comming in fourhead or can be used to detect the pattern of hairs in forehead.
   }
}

// this was the multiple detection of y coordinates of forehead line problem. Now it is solved after costing few miliseconds of times.
/*if(fw_points.size()!=2){
	cout<<"\n [x]error in forehead calculation[Multiple y coordinate point or singleton point is observed.]. \n Please avoid hairs over forehead.";
	fw=-1;
}
*/

if(fwleft==-1||fwright==1000){
	cout<<"\n [x]error in forehead calculation[Multiple y coordinate point or singleton point is observed.]. \n Face forehead estimation failed [Rare error].";
	fw=-1;
	false_fw_detection++;
}
else{
	
	fw= std::abs(fwleft - fwright);
	
}
cout<<"\nh= "<<h;
cout<<"\ncw= "<<cw;	
cout<<"\njw= "<<jw;
cout<<"\nfw= "<<fw;
cout<<"\n   ratio: ";
cout<<"\ncw:jw= "<<cw/(double)jw;
cout<<"\nfw:jw= "<<fw/(double)jw;
/*
meacj=+cw/(double)jw;
meafj=+fw/(double)jw;
cout<<"\nmeans:\n**********\ncj   =="<<meacj/(k+1)<<"      fj   =="<<meafj/(k+1);
*/

//test the person
face person(fw,jw,cw,h,cw);
//////////////////////////////

cout<<"\n Shape of face: "<<person.shape_name();


imshow("Original Image",cameraFeed);
waitKey(50);
}
}
	catch(cv::Exception e){
		cout<<e.msg;
		
	}
//system("pause");
}
cout<<"\n [x] false detection: "<<false_fw_detection<<"\n [x] false facedetection: "<<face_problem_count;
system("pause");
return 0;
}