#ifndef SYNTH_CIRCLE_PROJECTIONS_GENERATOR_H_
#define SYNTH_CIRCLE_PROJECTIONS_GENERATOR_H_

#include <opencv2/opencv.hpp>
#include <ceres_calibration/ceres_intrinsic_calib.h>

using std::cout;
using std::endl;


/**
 * @brief Generate synthetic images of an asymmetric circle grid pattern
 * 
 */
class SynthCircleProjectionsGenerator {

public:

    /**
     * @brief Construct a new Synth Circle Projections Generator object
     * 
     * @param shape       shape of the pattern (number of points in x and y direction)
     * @param square_size distance of points in the pattern
     * @param image_size  size of the image to project the pattern into
     * @param fx focal length in x direction
     * @param fy focal length in y direction (same as fx if -1)
     * @param cx center of projection in x direction (image_size.width/2 if -1)
     * @param cy center of projection in y direction (image_size.height/2 if -1)
     * @param distortion optional 1x5 matrix of distortion coefficients
     */
    SynthCircleProjectionsGenerator(cv::Size pattern_shape, double square_size, cv::Size image_size, double fx = 1000., double fy = -1, double cx = -1, double cy = -1, cv::Mat distortion = cv::Mat()): 
        shape(pattern_shape), square_size(square_size), image_size(image_size), fx(fx), fy(fy), cx(cx), cy(cy), distortion(distortion) {

        if (fy == -1) this->fy = fx;
        if (cx == -1) this->cx = image_size.width/2;
        if (cy == -1) this->cy = image_size.height/2;
    }

    /**
     * @brief Set the distortion for the simulated camera
     * 
     * @param distortion 1x5 matrix of distortion coefficients
     */
    void set_distortion(cv::Mat distortion){
        assert(distortion.cols == 1);
        assert(distortion.rows == 5);  
        this->distortion = distortion.clone();
    }

    void create_captures(size_t capture_cnt, std::filesystem::path output_dir = "/tmp/data");


private:

    cv::Mat distortion = cv::Mat();

    double fx, fy, cx, cy;
    cv::Size shape;
    cv::Size image_size;
    double square_size;
};


#endif