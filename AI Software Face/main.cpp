/*By Cebastian Santiago
* AI Software Face Detection
*/
#include <opencv2/core/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/videoio.hpp>
#include<iostream>
#include<string>
#include<fstream>
#include<vector>


using namespace cv;
using namespace face;


void read_photos( std::vector<Mat>& images, std::vector<int>& labels) {
	//read the photos into the vector
	//1st person
	std::string imagepath = samples::findFile("C:\\Users\\Cebas\\source\\repos\\AI Software Face\\AI Software Face\\Cebastian_20_20_70_70.jpg");
	images.push_back(imread(imagepath, IMREAD_GRAYSCALE));
	labels.push_back(0);
	
	imagepath = samples::findFile("C:\\Users\\Cebas\\source\\repos\\AI Software Face\\AI Software Face\\Cebastian1_20_20_70_70.jpg");
	images.push_back(imread(imagepath, IMREAD_GRAYSCALE));
	labels.push_back(0);

	//2nd person
	imagepath = samples::findFile("C:\\Users\\Cebas\\source\\repos\\AI Software Face\\AI Software Face\\NOE1_20_20_70_70.jpg");
	images.push_back(imread(imagepath, IMREAD_GRAYSCALE));
	labels.push_back(1);

	imagepath = samples::findFile("C:\\Users\\Cebas\\source\\repos\\AI Software Face\\AI Software Face\\NOE2_20_20_70_70.jpg");
	images.push_back(imread(imagepath, IMREAD_GRAYSCALE));
	labels.push_back(1);
}



void Face_Detection(cv::Mat& frames,cv::VideoCapture& video,CascadeClassifier& classifer,int width , int height, cv::Ptr<cv::face::FisherFaceRecognizer> model){
	//vector of string to produce name output and string to produce name
	std::vector<std::string> names = {"Noe","Cebastian"};
	std::string who = "";
	
	//clone the current frame
	for (;;) {
		video >> frames;
		Mat orginal = frames.clone();

		//conver the current frame to gray
		cv::Mat gray;
		cv::cvtColor(orginal, gray, cv::COLOR_BGR2GRAY);


		//find the current faces in the frame
		std::vector<cv::Rect_<int>> Faces;
		classifer.detectMultiScale(gray, Faces);


		//make the predection of who is in the video
		for (int i = 0; i < Faces.size(); i++) {
			 //process face by face
			Rect faces_i = Faces[i];
			//cropy the face form the image
			Mat face = gray(faces_i);

			//resize the image 
			Mat face_resized;
			cv::resize(face, face_resized, Size(width,height),2.0,2.0,INTER_CUBIC);

			//PREDICT WHO EVER IS IN THE VIDEO
			int predict = model->predict(face_resized);


			//output a rectangle to the video 
			rectangle(orginal, faces_i, CV_RGB(0,255,0), 1);
			
			who = format("You are ");
			if (predict >= 0 || predict <= 1) {
				who.append(names[predict]);
			}
			else { who.append("Unkown"); }

			//calculate where the txt should be printed
			int pos_x = std::max(faces_i.tl().x - 10, 0);
			int pos_y = std::max(faces_i.tl().y - 10, 0);

			//put the text in the image
			cv::putText(orginal, who, cv::Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 2.0,CV_RGB(0,255,0), 2.0);

		}


		//show the output
		cv::imshow("Face ID", orginal);

		//wait 20 Seconds:
		char key = (char)waitKey(20);
		// Exit this loop on escape:
		if (key == 27)
			break;
	}
}





int main(int argc, char* argv[]) {
	//variables and two vectors for lables
	cv::VideoCapture video(0);
	cv::Mat  image, frame , gray;
	std::vector<Mat> images;
	std::vector<int> labels;

	//facial recoginition class
	CascadeClassifier cascade, nestedCascade;
	double scale = 0;
	int s = 0;

	//path to the file
	std::string face = "C:\\opencv-4.1.1\\opencv\\install\\etc\\haarcascades\\haarcascade_frontalface_default.xml";
	
	//file name for data and fstream to read file in 
	std::ifstream inputstream;

	//check for two arguments
	if (argc != 2) {
		std::cout << "usage: " << argv[1] << "no ext file provided\n";
		exit(1);
	}

	inputstream.open(argv[1], std::ios::in); //open for reading the file input 

	//verfiy camera can be opened
	if (!video.isOpened()){
		std::cout << "Could not open camera" << std::endl;
		exit(1);
	}

	//call the read_cvs to read the data
	read_photos(images, labels);

	//get the width and height form the image
	int width = images[0].rows;
	int height = images[0].cols;
	

	// Create a FaceRecognizer and train it on the given images
	cv::Ptr<cv::face::FisherFaceRecognizer> model = FisherFaceRecognizer::create();
	model->train(images, labels);
	cascade.load(face);



	//check if the camera is open
	if (!video.isOpened()){
		std::cout << "We could not open the camera " << std::endl;
		exit(1);
	}


	//open the camera and read the image
	while (video.read(image)){
		
		//if we didnt recieve anything form the video
		if (image.empty()){
			std::cout << "Did not read anything form the video " << std::endl;
			exit(1);
		}
		
		
		//call the face_dectection algorithm function
		Face_Detection(gray,video,cascade,width,height,model);
		
		if (cv::waitKey(10) > 0) {
			break;
		}

	}
	video.~VideoCapture();
	return 0;
}






















