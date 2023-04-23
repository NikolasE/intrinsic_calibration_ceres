

#include <iostream>
#include <random>

#include <ceres_calibration/synth_circle_projections_generator.h>


using namespace std;
namespace fs = std::filesystem;



double rand_between( double low, double high )
{
    return ( (double)rand() * ( high - low ) ) / (double)RAND_MAX + low;
}



void SynthCircleProjectionsGenerator::create_captures(size_t capture_cnt, std::filesystem::path output_dir)
{

    cout << "writing to: " << output_dir << endl;
    cout << "creating " << capture_cnt << " images" << endl;

    fs::create_directory(output_dir);

    // create to get acces to metric pattern positions
    OpenCVAsymmetricCircleGridCalibration calib("/tmp");
    calib.initialize_metric_pattern(shape, square_size);

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);  
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;

    cout << "fx: " << fx << endl;
    cout << "fy: " << fy << endl;
    cout << "cx: " << cx << endl;
    cout << "cy: " << cy << endl;


    int img_cnt = 0;

    fs::path info_file_name = output_dir / "info.yml";
    cv::FileStorage info_file(info_file_name.string(), cv::FileStorage::WRITE);

    info_file << "fx" << fx;
    info_file << "fy" << fy;
    info_file << "cx" << cx;
    info_file << "cy" << cy;
    info_file << "distortion" << distortion;
    info_file << "capture_cnt" << int(capture_cnt);
    info_file << "image_size" << image_size;
    info_file << "pattern_shape" << shape;
    info_file << "square_size" << square_size;
    info_file << "metric_pattern_points" << calib.metric_pattern_points;

    while(img_cnt < capture_cnt)
    {

        cv::Mat image = cv::Mat::zeros(image_size, CV_8UC3);
        image.setTo(cv::Scalar(255, 255, 255));
        // cv::Mat image_gray = cv::Mat::zeros(image_size, CV_8UC1);

        // create a random rotation
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);

        // values from real captured images (or start from rpy)
        rvec.at<double>(0) = rand_between(-0.5, 0.5);
        rvec.at<double>(1) = rand_between(-0.5, 0.5);
        rvec.at<double>(2) = rand_between(0.9, 2.2);

        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
        tvec.at<double>(0) = rand_between(-0.2, 0.2);
        tvec.at<double>(1) = rand_between(-0.2, 0.2);
        tvec.at<double>(2) = rand_between(0.2, 0.8);


        std::vector<cv::Point2f> image_points;
        cv::projectPoints(calib.metric_pattern_points, rvec, tvec, K, distortion, image_points);

        // check if all points are inside the image
        bool all_inside = true;
        for (size_t i = 0; i < image_points.size(); i++)
        {
            const auto*p = &image_points[i];
            if (p->x < 0 || p->x >= image_size.width || p->y < 0 || p->y >= image_size.height)
            {
                all_inside = false;
                break;
            }
        }

        if (!all_inside)
        {
            continue;
        }


        cout << "image " << img_cnt << " / " << capture_cnt << endl;


        // draw the pattern into the image
        // cv::drawChessboardCorners(image, shape, image_points, true);

        // get distance between first two points to scale circles
        double dist = cv::norm(image_points[0] - image_points[1]);


        for (size_t i = 0; i < image_points.size(); i++)
        {
            const auto*p = &image_points[i];
            cv::circle(image, *p, dist/4, cv::Scalar(0, 0, 0), -1);
        }


        string prefix = "image_" + std::to_string(img_cnt);

        // write poses and features with opencv:

        info_file << prefix + "_rvec" << rvec;
        info_file << prefix + "_tvec" << tvec;
        info_file << prefix + "_image_points" << image_points;

        fs::path p = output_dir / (prefix + ".png");    
        cv::imwrite(p.string(), image);

        img_cnt++;          
    }

    info_file.release();

}



int main()
{
    SynthCircleProjectionsGenerator generator(cv::Size(4, 15), 0.035, cv::Size(1280, 720));

    // cv::Mat distortion = cv::Mat::zeros(5, 1, CV_64F);
    // distortion.at<double>(0) = 0.5;
    // generator.set_distortion(distortion);

    generator.create_captures(30, "/tmp/synth");
}