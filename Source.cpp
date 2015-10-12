// SERIKE CAKMAK
// scakmak13@ku.edu.tr
// 20131565

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat readImg(String file) {
	Mat source_img = imread(file);
	if( source_img.data ) { 
		return source_img;	
	}
}

Mat yccConv(Mat img)  {
	// RGB to YCrCb conversion
	// YCbCr is not an absolute color space; rather, it is a way of encoding RGB information. (Wiki)
	Mat ycc_img = Mat(img.size(),img.type());
	cvtColor(img, ycc_img, CV_BGR2YCrCb);
	return ycc_img;
}

Mat myHist(Mat img) {

	Mat res;            
	Mat	histogram = Mat::zeros(1, 256, CV_32SC1);
	int rows = img.rows;
	int cols = img.cols;

	// Calculating histogram
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			histogram.at<int>(img.at<Vec3b>(i,j)[0]) += 1;
		}
	}

	// h max value
	int h = 0;
	for (int j = 0; j < 255; j++) {
		if(histogram.at<int>(j) > h)
			h = histogram.at<int>(j);
	}

	res = Mat::ones(256, 256, CV_8UC3);
	rows = res.rows;
	// Drawing 
	for (int j = 0 ; j < 255; j++) {
		line(res, 
			Point(j, rows), 
			Point(j, rows - (histogram.at<int>(j) * rows/h)), 
			Scalar(255,128,255), 1, 8, 0);
	}
	return res;
}

Mat calculateHis(Mat hist) {

	int histSize = 256; //from 0 to 255
	// Set the ranges ( for B,G,R) ) since default is BGR for imread and it is kept that way
	float range[] = {0, 256};
	const float* histRange = {range};
	bool uniform = true; 
	bool accumulate = false;

	//Split Y Cb Cr planes
	vector<Mat> ycc_planes;
	split(hist, ycc_planes);

	Mat y_hist;
	//Calculating Intensity Histogram  (YCbCr => Y is the intensity)
	calcHist( &ycc_planes[0], 1, 0, Mat(), y_hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );
	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	// Normalize the result to [ 0, histImage.rows ]
	normalize(y_hist, y_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	// Draw histogram for intensity channel
	for(int i=1; i<histSize; i++) {
		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(y_hist.at<float>(i-1))),
			Point(bin_w*(i), hist_h - cvRound(y_hist.at<float>(i))),
			Scalar(255, 128, 255), 2, 8, 0  );
	}

	return histImage;
}

Mat gammaCorrect(String name) {
	//////////////////////////Correct Gamma/////////////////////////// 
	int gamma = 2.0;
	cout<<"Please provide a gamma value for gamma correction to be performed!"<<endl;
	cout<<"You can enter 2 for using the default value!"<<endl;
	cin>> gamma	;

	double inv_gamma = 1.0/gamma;

	Mat lut_mat(1, 256, CV_8UC1 );
	uchar * p = lut_mat.ptr();
	for( int i=0; i<256; i++ ) {
		p[i] = (int)(pow((double) i/255.0,inv_gamma)*255.0);
	}

	Mat result;
	LUT(readImg(name), lut_mat, result);

	return result;
}

Mat equalize(Mat ycc) {

	//Split Y Cb Cr planes
	vector<Mat> ycc_planes;
	split(ycc, ycc_planes);

	equalizeHist(ycc_planes[0], ycc_planes[0]); //equalize histogram on the 1st channel (Y)
	Mat equalized_ycc;
	merge(ycc_planes,equalized_ycc); //merge 3 channels including the 1st channel 
	Mat equalized;
	cvtColor(equalized_ycc, equalized, CV_YCrCb2BGR); //convert from YCrCb to BGR format 

	return equalized;
}

Mat myEqu(Mat ycc) {
	Mat res;
	int	histogram[256];
	for(int i=0; i<256; i++) {
		histogram[i] = 0;  // inital value zero
	}

	int rows = ycc.rows;
	int cols = ycc.cols;
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			histogram[ycc.at<Vec3b>(i,j)[0]] += 1; // count intensity values
		}
	}

	double poss[256];
	for(int i=0; i<256; i++) {
		poss[i] = (double) histogram[i] / (rows*cols); // probability values
	}

	int cumulative[256];
	cumulative[0] = histogram[0];
	for(int i=1; i<256; i++) {
		cumulative[i]= histogram[i] + cumulative[i-1]; // cumulative calculation
	}

	double scale = (double) 255/(rows*cols);
	int skvalues[256];
	for(int i=0; i<256; i++) {
		skvalues[i] = cvRound((double)cumulative[i] * scale); // scaling (finding Sk values)
	}

	double sk2[256];
	for(int i=0; i<256; i++) {
		sk2[i]=0;
	}

	for(int i=0; i<256; i++) {
		sk2[skvalues[i]] += poss[i]; // sk2 values for histogram
	}

	int equhist[256];
	for(int i=0; i<256; i++)
		equhist[i] = cvRound(sk2[i]*255); // equalized hist values

	Mat equimage = ycc.clone();
	for(int y=0; y<rows; y++) {
		for(int x=0; x< cols; x++) {
			equimage.at<Vec3b>(y,x)[0] = saturate_cast<uchar>(skvalues[ycc.at<Vec3b>(y,x)[0]]); // equalized image
		}
	}
	// equalized and back bgr converted image
	cvtColor(equimage, res , CV_YCrCb2BGR); //convert from YCrCb to BGR format 

	// h max value in equ hist
	int h = 0;
	for (int j=0; j<255; j++) {
		if(equhist[j]>h)
			h = equhist[j];
	}

	Mat resh = Mat::ones(326, 512, CV_8UC3);
	rows = resh.rows;
	// Drawing 
	for (int j=0; j<255; j++) {
		line(resh, 
			Point(j,rows), 
			Point(j,rows - (equhist[j]*rows/h)), 
			Scalar(255,128,255), 1, 8, 0);
	}

	namedWindow("My Equalized Histogram", CV_WINDOW_AUTOSIZE);
	imshow("My Equalized Histogram", resh);

	return res;
}

void helper (Mat help,int skvalues[]) {
	int	histogram[256];
	for(int i = 0; i < 256; i++) {
		histogram[i] = 0;  // inital value zero
	}

	int rows = help.rows;
	int cols = help.cols;
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			histogram[help.at<Vec3b>(i,j)[0]] += 1; // count intensity values
		}
	}

	double poss[256];
	for(int i=0; i<256; i++) {
		poss[i] = (double) histogram[i] / (rows*cols); // probability values
	}

	int cumulative[256];
	cumulative[0] = histogram[0];
	for(int i=1; i<256; i++) {
		cumulative[i]= histogram[i] + cumulative[i-1]; // cumulative calculation
	}

	double scale = (double) 255/(rows*cols);

	for(int i=0; i< 256; i++) {
		skvalues[i] = cvRound((double)cumulative[i] * scale); // scaling (finding Sk values)
	}

}

Mat localAdpt(Mat img) {
	Mat res;

	Mat equimage = img.clone();
	int size = 5;
	Mat M = Mat::ones(size,size,img.type()) ;

	for(int x=0;x<img.rows-size;x++) {
		for(int y=0;y<img.cols-size;y++) {

			int skvalues[256];
			helper(M,skvalues);

			for(int i=0;i<size;i++){
				for(int j=0;j<size;j++) {
					M.at<Vec3b>(i,j)[0]=img.at<Vec3b>(i+x,j+y)[0];
					equimage.at<Vec3b>(x,y)[0] = saturate_cast<uchar>(skvalues[M.at<Vec3b>(i,j)[0]]); // equalized image
				}
			}	
		}	
	}

	cvtColor(equimage, res , CV_YCrCb2BGR); //convert from YCrCb to BGR format 

	return res;
}

int main()
{
	String name = "cor_flower3.jpg";
	cout<< "Please provide the name of the JPEG image to load"<<endl;
	cin>> name;

	namedWindow("Original image", CV_WINDOW_AUTOSIZE);
	imshow("Original image", readImg(name));	
	Mat ycc =  yccConv(readImg(name));
	//imshow("YCbCr converted image",ycc); 

	///////////////// INTENSITY HIST OPENCV ////////////////
	namedWindow("Intensity Histogram OpenCv", CV_WINDOW_AUTOSIZE );
	imshow("Intensity Histogram OpenCv", calculateHis(ycc));

	///////////////////// MY IMPLEMENTATION /////////////////
	namedWindow("My Intensity Histogram", CV_WINDOW_AUTOSIZE );
	imshow("My Intensity Histogram", myHist(ycc));

	////////////////////  GAMMA CORRECTION //////////////////
	Mat gamma_corrected = gammaCorrect(name);
	imshow("Gamma Corrected Image", gamma_corrected);
	imshow("Gamma Corrected Histogram", calculateHis(yccConv(gamma_corrected)));
	// save gama corrected image
	imwrite( name+"GammaCorrected.jpg", gamma_corrected );

	////////////////////////MY EQUALIZATION////////////////////
	namedWindow("My Histogram Equalized Image", CV_WINDOW_AUTOSIZE);
	imshow("My Histogram Equalized Image", myEqu(ycc));
	// save image
	imwrite( name+"MyHisEqualized.jpg", myEqu(ycc) );

	///////////////HISTOGRAM EQUALIZATION//////////////////////
	namedWindow("Histogram Equalized Image OpenCv", CV_WINDOW_AUTOSIZE);
	imshow("Histogram Equalized Image OpenCv", equalize(ycc));
	// save image
	imwrite( name+"HistogramEqualized.jpg", equalize(ycc) );

	///////////////////// LOCAL ADAPTIVE ///////////////////////
	Mat la = localAdpt(ycc);
	namedWindow("Local Adaptive", CV_WINDOW_AUTOSIZE );
	imshow("Local Adaptive", la);
	namedWindow("Local Adaptive Histogram OpenCv", CV_WINDOW_AUTOSIZE );
	imshow("Local Adaptive Histogram OpenCv", calculateHis(la));
	imwrite( name+"LocalAdaptive.jpg", la );

	cvWaitKey(0);
	return 0;
}