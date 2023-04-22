#ifndef CERES_INTRINSIC_CALIB_H
#define CERES_INTRINSIC_CALIB_H

#include <iostream>
#include <filesystem>

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
    double initial_error;   // reprojection error after computing initial pose with guessed focal length and no distortion
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


    bool run_optimization(bool optimize_distortion = true);

    /**
     * @brief pattern points in metric coordinates and in the pattern's coordinate system
     *  for a planar pattern, all z-coordinates are 0. Values are in meters.
     * 
     */
    std::vector<cv::Point3f> metric_pattern_points; 


    /**
     * @brief Creates a visualization of the projected points (initial and optimized)
     *       and writes it to the output directory.
     * 
     * @param cap Capture to visualize
     * @return true if debug image was written to 'output_dir'
     */
    bool visualize_projections(const Capture& cap);


    /**
     * @brief Runs opencv's calibration function using K_initial and the Capture's initial pose to 
     *        compare the results to the Ceres-Optimization. 
     * 
     * @param optimize_distortion if true, distortion coefficients are optimized as well
     * @return double mean reprojection error after opencv calibration
     */
    double opencv_calibrate_camera(bool optimize_distortion = false);


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


    /**
     * @brief Collection of all captures. Each with one image and a corresponding observation
     * 
     */
    std::vector<Capture> captures;

    
    /**
     * @brief Find pattern in the Capture's image and store the extracted points in cap.observed_points. 
     *        This function depends on the used pattern type and is implemented in the derived classes.
     * 
     * @param cap Capture to extract pattern from
     * @return true iff pattern was detected
     */
    virtual bool extract_pattern(Capture& cap) = 0;

    /**
     * @brief Collect all images in the given directory and extract the pattern from each image.
     *        The extracted points are stored in the corresponding Capture's observed_points.
     *        The first image is used to initialize the focal length by evaluating multiple values and 
     *        choosing the one with the lowest reprojection error.
     *        Before this function is called, the pattern's metric points must be set.
     * 
     * @param img_directory directory containing images
     * @param num_images_in_dir if not nullptr, the number of images in the directory is stored here
     * @return double mean reprojection error after computing initial pose with guessed focal length and no distortion
     */
    double collect_pattern_views(const std::filesystem::path& img_directory, size_t* num_images_in_dir = nullptr);

    /**
     * @brief Compute initial pose of pattern in the given capture using the initial guess for the focal length and
     *        no distortion by calling cv::solvePnP.
     * 
     * @param cap Capture to compute pose for
     * @return double mean reprojection error after computing initial pose with guessed focal length and no distortion
     */
    double get_initial_pose(Capture& cap);

    /**
     * @brief Mean reprojection error over all Capture's after computing initial pose with guessed focal length and no distortion 
     * 
     */
    double initial_mean_error = 0.0;

    /**
     * @brief Mean reprojection error over all Capture's after full optimization 
     * 
     */
    double optimized_mean_error = 0.0;

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


    // implementation for update_initial_error and update_optimized_error
    /**
     * @brief Helper function to compute reprojection error for a capture
     * 
     * @param cap Capture to compute error for
     * @param initial flag to indicate whether to use initial or optimized pose
     * @param cam_matrix K-matrix to use
     * @param dist_coeffs optional distortion coefficients
     * @return double mean squared reprojection error
     */
    double get_error(Capture& cap, bool initial, cv::Mat cam_matrix, const cv::Mat dist_coeffs = cv::Mat());

private:
 
    /**
     * @brief Ceres problem to optimize
     * 
     */
    ceres::Problem problem;


    /**
     * @brief Helper function to set the bounds for a parameter with a given deviation ratio
     * 
     * @param params parameter array
     * @param index  index of parameter to set bounds for
     * @param ratio  ratio of parameter value to use as deviation, relative to current value
     */
    void setParameterBounds(double* params, size_t index, double ratio) {
        double min = params[index] * (1.0 - ratio);
        double max = params[index] * (1.0 + ratio);
        problem.SetParameterLowerBound(params, index, min);
        problem.SetParameterUpperBound(params, index, max);
    }


    /**
     * @brief Helper function to set min and max bounds for a parameter
     * 
     * @param params Parameter array
     * @param index index of parameter to set bounds for
     * @param min   new minimal value of parameter
     * @param max   new maximal value of parameter
     */
    void setParameterBounds(double* params, size_t index, double min, double max) {
        problem.SetParameterLowerBound(params, index, min);
        problem.SetParameterUpperBound(params, index, max);
    }


};


/**
 * @brief Intrinsic calibration for a flat asymmetric OpenCV CircleGrid pattern.
 * 
 */
class OpenCVAsymmetricCircleGridCalibration: public IntrinsicCalibration 
{
public:

    /**
     * @brief Construct a new Open C V Asymmetric Circle Grid Calibration object. Directly initializes the metric pattern points.
     * 
     * @param output_directory Output directory for visualizations and results
     * @param pattern_shape Shape of pattern (width, height)
     * @param square_size    Distance in meters between centers of circles in same row or column [@see initialize_metric_pattern]
     */
    OpenCVAsymmetricCircleGridCalibration(const std::filesystem::path& output_directory, cv::Size pattern_shape, double square_size):
        IntrinsicCalibration(output_directory),
        pattern_shape(pattern_shape),
        square_size(square_size)
    {
        initialize_metric_pattern(pattern_shape, square_size);
    };


    /**
     * @brief Construct a new Open C V Asymmetric Circle Grid Calibration object. Does not initialize the metric pattern points.
     *        This must be done manually by calling initialize_metric_pattern.
     * 
     * @param output_directory Output directory for visualizations and results
     */
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
    
    /**
     * @brief Extract pattern points from the given capture by calling cv::findCirclesGrid(...,cv::CALIB_CB_ASYMMETRIC_GRID)
     * 
     * @param cap 
     * @return true iff pattern was detected
     */
    bool extract_pattern(Capture& cap) override;

    /**
     * @brief Visualize detection of a pattern on the given image. 
     *        Writes image to output_dir than contains the capture's filename.
     * 
     * @param cap Visualized Capture
     * @return true if debug image was written to 'output_dir'
     */
    bool visualize_detection(const Capture& cap) override;


    cv::Size pattern_shape; // shape of pattern (width, height)
    double square_size;     // distance in meters between centers of circles in same row or column
};


#endif