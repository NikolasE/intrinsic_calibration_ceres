#ifndef CERES_INTRINSIC_CALIB_H
#define CERES_INTRINSIC_CALIB_H

#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "ceres_intrinsic_calib.h"

// This error term describes the view of a pattern from a camera.
// using the current estimated relative pose between the camera and the pattern
// as well as the intrinsic calibration (including distortion parameters), 
// the reprojection error of all points in the pattern is computed.
// The return value is the mean reprojection error of all points.
// TODO: what happens if we use the maximum?
struct PatternViewReprojectionError {

    // The metric_pattern_points are given in the local frame of the pattern. 
    PatternViewReprojectionError(const std::vector<cv::Point3f>& metric_pattern_points,
                             const std::vector<cv::Point2f>& image_points,
                             uint width, uint height)
        : metric_pattern_points(metric_pattern_points), 
          measured_image_points(image_points),
          width(width), height(height)
          {
            assert(metric_pattern_points.size() == image_points.size());
            assert(metric_pattern_points.size() > 0);
        }

    template <typename T>
    bool operator()(const T* pattern_pose, // brings pattern points to camera frame
                    const T* camera_intrinsics, // fx, fy, cx, cy, [k1, k2, p1, p2, k3]
                    T* error) const 
    {

        // std::cout << "operator called" << std::endl;
        T sum_of_squares = T(0);

        // use normalized camera intrinsics to have all parameters around +- 1
        const T& fx = camera_intrinsics[0]*T(width);
        const T& fy = camera_intrinsics[1]*T(height);
        const T& cx = camera_intrinsics[2]*T(width);
        const T& cy = camera_intrinsics[3]*T(height);


        // std::cout << "fx: " << fx << std::endl;
        // std::cout << "fy: " << fy << std::endl;
        // std::cout << "cx: " << cx << std::endl;
        // std::cout << "cy: " << cy << std::endl;

        // const T& k1 = camera_intrinsics[4];
        // const T& k2 = camera_intrinsics[5];
        // const T& p1 = camera_intrinsics[6];
        // const T& p2 = camera_intrinsics[7];
        // const T& k3 = camera_intrinsics[8];

        // std::cout << "in_cam_frame: " << pattern_pose[0] << " " << pattern_pose[1] << " " << pattern_pose[2] << std::endl;
        // std::cout << "trans " << pattern_pose[3] << " " << pattern_pose[4] << " " << pattern_pose[5] << std::endl;

        // T sum_x = T(0);
        // T sum_y = T(0);


        for (int i = 0; i < metric_pattern_points.size(); ++i) {

            const auto& p = metric_pattern_points[i];

            T p3d[3] = {T(p.x), T(p.y), T(p.z)}; 
            T in_cam_frame[3];
            ceres::AngleAxisRotatePoint(pattern_pose, p3d, in_cam_frame);
            in_cam_frame[0] += pattern_pose[3];
            in_cam_frame[1] += pattern_pose[4];
            in_cam_frame[2] += pattern_pose[5];


            // if (pattern_pose[5] < 0)
            // {
            //     std::cout << "z negative " << pattern_pose[5] << std::endl; 
            // }

            T x = in_cam_frame[0] / in_cam_frame[2];
            T y = in_cam_frame[1] / in_cam_frame[2];

            // T r2 = x*x + y*y;
            // T r4 = r2*r2;
            // T r6 = r4*r2;
            // T radial_distortion = 1.0 + k1*r2 + k2*r4 + k3*r6;
            // T x_distorted = x * radial_distortion + 2.0*p1*x*y + p2*(r2 + 2.0*x*x);
            // T y_distorted = y * radial_distortion + 2.0*p2*x*y + p1*(r2 + 2.0*y*y);

            // T u = fx * x_distorted + cx;
            // T v = fy * y_distorted + cy;


            // optimization without distortion
            T u = fx * x + cx;
            T v = fy * y + cy;

            // compute squared error
            T x_diff = u - T(measured_image_points[i].x);
            T y_diff = v - T(measured_image_points[i].y);

            T sq_error = x_diff * x_diff + y_diff * y_diff;
            error[i] = sq_error;
            // sum_of_squares += sq_error;
        }

        // error computes as in RunIntrinsicCalibration::get_error
        // *error = sum_of_squares / T(metric_pattern_points.size());

        // std::cout << "root mean_squared_error: " << *mean_squared_error << std::endl;
        return true;
    }

    static ceres::CostFunction* CreateNoDistortion(const std::vector<cv::Point3f>& metric_pattern_points,
                                       const std::vector<cv::Point2f>& image_points, uint width, uint height) {
        // 3+3 = 6 pose parameters (rvec, tvec)
        // 9 intrinsic parameters (fx, fy, cx, cy, k1, k2, p1, p2, k3)
        // 15*4: reprojection error per feature
        return (new ceres::AutoDiffCostFunction<PatternViewReprojectionError, 15*4, 6, 4>( // 1,6,9
            new PatternViewReprojectionError(metric_pattern_points, image_points, width, height)));
    }

    const std::vector<cv::Point2f>& measured_image_points;
    const std::vector<cv::Point3f>& metric_pattern_points;

private:
    
    uint width, height; // image size. Could be static to share, but in future we could support different image sizes
    std::vector<cv::Point2f> projected; // reuse memory
};



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



class RunIntrinsicCalibration {
public:
    RunIntrinsicCalibration(const std::filesystem::path& img_directory,
                            uint pattern_width = 15, uint pattern_height = 4,
                            double square_size = 0.035);

    cv::Size pattern_shape;
    double square_size;
// private:

    std::vector<cv::Point3f> metric_pattern_points; // same flat (z=0) pattern for all images


    void visualize_projections(const Capture& cap);


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

    std::filesystem::path debug_directory;

    std::vector<Capture> captures;

    int width, height;

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

    static std::vector<cv::Point3f> init_flat_asymmetric_pattern(cv::Size shape, double square_size);
    bool extract_pattern(Capture& cap);

    double collect_pattern_views(const std::filesystem::path& img_directory, size_t* num_images_in_dir = nullptr);

    double get_initial_pose(Capture& cap);

    cv::Mat K_initial; // initial guess for intrinsic parameters
    cv::Mat K_optimized; // optimized intrinsic parameters
    cv::Mat dist_coeffs_optimized; // optimized distortion coefficients (we currently have no initial guess for these)
};


#endif