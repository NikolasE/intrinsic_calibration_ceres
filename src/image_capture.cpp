// capture images with OpenCV from Webcam and as soon as the user cliks on the image, write in in a file


#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;


int main(){
    cv::VideoCapture cap(0);
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    int cnt = 0;

    while(1){
        cv::Mat frame;
        cap >> frame;

        cv::flip(frame, frame, 1);


        cv::imshow("Frame", frame);
        char c = (char)cv::waitKey(2);
        std::cout << int(c) << std::endl;
        if(c==27)
        {
            break;
        }
        if (int(c) < 0)
            continue;

        std::cout << "saving image" << std::endl;
        cv::imwrite("calib_images/image" + std::to_string(cnt) + ".png", frame);
        cnt++;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}