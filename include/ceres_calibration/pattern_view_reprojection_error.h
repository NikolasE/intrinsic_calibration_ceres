#ifndef CERES_CALIBRATION_PATTERN_VIEW_REPROJECTION_ERROR_H
#define CERES_CALIBRATION_PATTERN_VIEW_REPROJECTION_ERROR_H

#include <vector>

#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

/**
 * @brief The PatternViewReprojectionError struct can be used in a ceres::CostFunction
 *        to compute the reprojection error of a pattern view. For a given pattern pose and
 *        camera intrinsics, the error is the difference between the measured and the projected points. 
 *        The pattern pose is given as a 6D vector (3D Roditues rotation and 3D translation) and the camera intrinsics
 *        are given as a 5D vector (fx, fy, cx, cy, [k1, k2, p1, p2, k3]). The distortion is only used if optimize_distortion is true.
 *        fx, fy, cx, cy are normalized to be in the range [0, 1] and are scaled by the image width and height in the constructor.
 *        The error vector has the same size as the number of pattern points. For each point, the squared error is computed  
 *        This class supports only no or five distortion parameters (openCV plum bob model)
 */
struct PatternViewReprojectionError {

    // The metric_pattern_points are given in the local frame of the pattern. (same definition as in OpenCV calibrateCamera)
    PatternViewReprojectionError(const std::vector<cv::Point3f>& metric_pattern_points,
                             const std::vector<cv::Point2f>& image_points,
                             uint width, uint height, bool optimize_distortion)
        : metric_pattern_points(metric_pattern_points), 
          measured_image_points(image_points),
          width(width), height(height), // image size, used to scale normalized intrinsics
          optimize_distortion(optimize_distortion)
        {
            assert(metric_pattern_points.size() == image_points.size());
            assert(metric_pattern_points.size() > 0);
        }

    bool optimize_distortion;

    template <typename T>
    bool operator()(const T* pattern_pose, // brings pattern points to camera frame
                    const T* camera_intrinsics, // fx, fy, cx, cy, [k1, k2, p1, p2, k3]
                    T* error) const 
    {
        // T sum_of_squares = T(0);

        // use normalized camera intrinsics to have all parameters around +- 1
        const T& fx = camera_intrinsics[0]*T(width);
        const T& fy = camera_intrinsics[1]*T(height);
        const T& cx = camera_intrinsics[2]*T(width);
        const T& cy = camera_intrinsics[3]*T(height);

        for (int i = 0; i < metric_pattern_points.size(); ++i) {

            const auto& p = metric_pattern_points[i];

            T p3d[3] = {T(p.x), T(p.y), T(p.z)}; 
            T in_cam_frame[3];
            ceres::AngleAxisRotatePoint(pattern_pose, p3d, in_cam_frame);
            in_cam_frame[0] += pattern_pose[3];
            in_cam_frame[1] += pattern_pose[4];
            in_cam_frame[2] += pattern_pose[5];

            T x = in_cam_frame[0] / in_cam_frame[2];
            T y = in_cam_frame[1] / in_cam_frame[2];

            T u, v;

            if (optimize_distortion) {
                const T& k1 = camera_intrinsics[4];
                const T& k2 = camera_intrinsics[5];
                const T& p1 = camera_intrinsics[6];
                const T& p2 = camera_intrinsics[7];
                const T& k3 = camera_intrinsics[8];

                T r2 = x*x + y*y;
                T r4 = r2*r2;
                T r6 = r4*r2;
                T radial_distortion = 1.0 + k1*r2 + k2*r4 + k3*r6;
                T x_distorted = x * radial_distortion + 2.0*p1*x*y + p2*(r2 + 2.0*x*x);
                T y_distorted = y * radial_distortion + 2.0*p2*x*y + p1*(r2 + 2.0*y*y);

                u = fx * x_distorted + cx;
                v = fy * y_distorted + cy;
            }else{
                u = fx * x + cx;
                v = fy * y + cy;
            }
            // compute squared error
            T x_diff = u - T(measured_image_points[i].x);
            T y_diff = v - T(measured_image_points[i].y);

            T sq_error = x_diff * x_diff + y_diff * y_diff;
            error[i] = sq_error;
            // sum_of_squares += sq_error;
        }

        // error computes as in IntrinsicCalibration::get_error
        // *error = sum_of_squares / T(metric_pattern_points.size());

        // std::cout << "root mean_squared_error: " << *mean_squared_error << std::endl;
        return true;
    }


    /**
     * @brief Create a ceres::CostFunction for the PatternViewReprojectionError
     * 
     * @param metric_pattern_points Pattern points in the local frame of the pattern (same definition as in OpenCV calibrateCamera)
     * @param image_points          Observed image points in this view
     * @param width                 Image width (for scaled intrinsics)
     * @param height                Image height
     * @param optimize_distortion   If true, the distortion parameters are optimized
     * @return ceres::CostFunction* 
     */
    static ceres::CostFunction* Create(const std::vector<cv::Point3f>& metric_pattern_points,
                                       const std::vector<cv::Point2f>& image_points, uint width, uint height, bool optimize_distortion) {
        PatternViewReprojectionError* error = new PatternViewReprojectionError(metric_pattern_points, image_points, width, height, optimize_distortion);
        // 3+3 = 6 pose parameters (rvec, tvec)
        // 4 intrinsic parameters (fx, fy, cx, cy), optionally 5 distortion parameters (k1, k2, p1, p2, k3)
        // Dynamic: reprojection error per feature
        // TODO: check DynamicAutoDiffCostFunction if also number of distortion parameters can be dynamic
        if (optimize_distortion) {
            return (new ceres::AutoDiffCostFunction<PatternViewReprojectionError, ceres::DYNAMIC, 6, 9>( 
                error, metric_pattern_points.size()));
        }else{
            return (new ceres::AutoDiffCostFunction<PatternViewReprojectionError, ceres::DYNAMIC, 6, 4>( 
                error, metric_pattern_points.size()));
        }

    }

private:
    const std::vector<cv::Point2f>& measured_image_points;
    const std::vector<cv::Point3f>& metric_pattern_points;
    const uint width, height; // image size. Could be static to share between instances, but in future we could support different image sizes
};


#endif