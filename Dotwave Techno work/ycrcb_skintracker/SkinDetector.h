#pragma once
#include<opencv\cv.h>
using namespace std;
class SkinDetector
{
public:
SkinDetector(void);
~SkinDetector(void);

cv::Mat getSkin(cv::Mat input);

private:
int Y_MIN;
int Y_MAX;
int Cr_MIN;
int Cr_MAX;
int Cb_MIN;
int Cb_MAX;
};

class face
{
	private:
		double fw;
		double jw;
		double cw;
		double h;
		double w;
		string shape;
	public:
		string calc_shape(face obj){
			/// constraints are required to change for correct results.
			obj.shape= (obj.fw==obj.jw)?(obj.jw==obj.cw)?(obj.h/obj.w>1.5)?"OBLONG":"SQUARE":(obj.jw<obj.cw)?(obj.h/obj.w>1.5)?"OVAL":"DIAMOND":"NOT DEFINED YET":(obj.fw<obj.jw && obj.jw<obj.cw)?"TRIANGLE":(obj.fw>obj.jw && obj.jw>obj.cw)?"HEART":"NOT DEFINED YET";
			return obj.shape;
		}
		face(double fw, double jw, double cw, double h, double w){
			 this->fw= fw;
			 this->cw= cw;
			 this->h= h;
			 this->jw= jw;
			 this->w= w;
		 this->shape= this->calc_shape(*this);
		 }
		string shape_name(){
			return shape;
		}
};
