   // error function if we can use doubles (numeric diff)
   
     cv::Matx33d K(camera_intrinsics[0].a, 0, camera_intrinsics[2].a),
                      0, double(camera_intrinsics[1]), double(camera_intrinsics[3]),
                      0, 0, 1);

        cv::Matx<double, 3, 1> rvec(pattern_pose[0], pattern_pose[1], pattern_pose[2]);              
        cv::Matx<double, 3, 1> tvec(pattern_pose[3], pattern_pose[4], pattern_pose[5]);
        cv::Matx<double, 5, 1> dist_coeffs(camera_intrinsics[4], camera_intrinsics[5], camera_intrinsics[6], camera_intrinsics[7], camera_intrinsics[8]);
        
        // transform pattern to camera frame, project into image plane, apply distortion
        cv::projectPoints(metric_pattern_points, rvec, tvec, K, dist_coeffs, projected);

        // compute mean reprojection error
        double sum = 0;
        for (int i = 0; i < projected.size(); ++i) {
            sum += cv::norm(projected[i] - measured_image_points[i]);
        }

        *mean_reprojection_error = T(sum / projected.size());