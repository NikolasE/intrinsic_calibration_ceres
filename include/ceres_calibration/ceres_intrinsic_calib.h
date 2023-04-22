#ifndef CERES_INTRINSIC_CALIB_H
#define CERES_INTRINSIC_CALIB_H

#include <iostream>
#include <filesystem>

// #include <opencv2/opencv.hpp>
// #include <ceres/ceres.h>
// #include <ceres/rotation.h>

#include "ceres_calibration/pattern_view_reprojection_error.h"



/**
 * @brief Represents a single image in which (up to) one calibration target is visible
 * 
 */
struct Capture {

    Capture(std::filesystem::path path):
        filename(path.stem().string())
    {}

    std::string filename; // filename without extension

    std::vector<cv::Point2f> observed_points; // extracted pattern points
    double initial_error; // reprojection error after computing initial pose with guessed focal length and no distortion
    double optimized_error; // reprojection error after full optimization

    cv::Mat rvec_initial, tvec_initial; // initial pose of pattern (corresponds to initial_error)
    cv::Mat rvec_opt = cv::Mat::zeros(3,1,CV_64F),
            tvec_opt = cv::Mat::zeros(3,1,CV_64F);

    cv::Mat img; // rgb image

    bool pattern_visible; // true iff pattern was detected on this image

    // keep track of projections for visualization
    std::vector<cv::Point2f> projected_initial; // projections for initial guess
    std::vector<cv::Point2f> projected_opt;    // projections for optimized parameters
};



class IntrinsicCalibration {
public:
    IntrinsicCalibration(const std::filesystem::path& output_directory);


    // cv::Size pattern_shape;
    // double square_size;



// private:


    bool run_optimization(bool optimize_distortion = false);


    std::vector<cv::Point3f> metric_pattern_points; // same flat (z=0) pattern for all images


    void visualize_projections(const Capture& cap);


    void opencv_calibrate_camera(bool optimize_distortion = false);


    /**
     * @brief Update initial error for a capture
     * 
     * @param cap computed error (also stored in cap.initial_error)
     */
    double update_initial_error(Capture& cap);


    /**
     * @brief Update optimized error for a capture
     * 
     * @param cap computed error (also stored in cap.optimized_error)
     */
    double update_optimized_error(Capture& cap);

    // implementation for update_initial_error and update_optimized_error
    double get_error(Capture& cap, bool initial, cv::Mat cam_matrix, const cv::Mat dist_coeffs = cv::Mat());


    std::vector<Capture> captures;


    ceres::Problem problem;

    void setParameterBounds(double* params, size_t index, double ratio) {
        double min = params[index] * (1.0 - ratio);
        double max = params[index] * (1.0 + ratio);
        problem.SetParameterLowerBound(params, index, min);
        problem.SetParameterUpperBound(params, index, max);
    }


    void setParameterBounds(double* params, size_t index, double min, double max) {
        problem.SetParameterLowerBound(params, index, min);
        problem.SetParameterUpperBound(params, index, max);
    }

    
    
    virtual bool extract_pattern(Capture& cap) = 0;

    double collect_pattern_views(const std::filesystem::path& img_directory, size_t* num_images_in_dir = nullptr);

    double get_initial_pose(Capture& cap);

    double initial_mean_error = 0.0;

    cv::Mat K_initial; // initial guess for intrinsic parameters
    cv::Mat K_optimized; // optimized intrinsic parameters
    cv::Mat dist_coeffs_optimized = cv::Mat::zeros(1, 5, CV_64F); // optimized distortion coefficients (we currently have no initial guess for these)

    uint width,  // image width
         height; // image height

    /**
     * @brief Visualize detection of a pattern on the given image. 
     *        Writes image to output_dir than contains the capture's filename.
     * 
     * @param cap Visualized Capture
     * @return true if debug image was written to 'output_dir'
     */
    virtual bool visualize_detection(const Capture& cap) = 0;

protected:
    std::filesystem::path output_dir;



};


class OpenCVAsymmetricCircleGridCalibration: public IntrinsicCalibration {


public:
    OpenCVAsymmetricCircleGridCalibration(const std::filesystem::path& output_directory, cv::Size pattern_shape, double square_size):
        IntrinsicCalibration(output_directory),
        pattern_shape(pattern_shape),
        square_size(square_size)
    {
        initialize_metric_pattern(pattern_shape, square_size);
    };

    OpenCVAsymmetricCircleGridCalibration(const std::filesystem::path& output_directory):
        IntrinsicCalibration(output_directory)
    {};

    /**
     * @brief Initialize pattern points for a flat asymmetric OpenCV CircleGrid pattern.
     *        Fills metric_pattern_points.
     * 
     * @param shape   shape of pattern (width, height)
     * @param square_size distance in meters between centers of circles in same row or column
     */
    void initialize_metric_pattern(cv::Size shape, double square_size);
    
    bool extract_pattern(Capture& cap) override;
    bool visualize_detection(const Capture& cap) override;


    cv::Size pattern_shape;
    double square_size;
};


#endif