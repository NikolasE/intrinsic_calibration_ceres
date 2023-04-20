#include "ceres_calibration/ceres_intrinsic_calib.h"
#include <iostream>

using namespace std;
namespace fs = std::filesystem;
#include <boost/range/combine.hpp>

RunIntrinsicCalibration::RunIntrinsicCalibration(const std::filesystem::path& img_directory,
                                  size_t pattern_width, size_t pattern_height,
                                  double square_size,
                                  double focal_length): 
                                  pattern_shape(pattern_width, pattern_height),
                                  square_size(square_size),
                                  initial_focal_length(focal_length) 
{
   metric_pattern_points = RunIntrinsicCalibration::init_flat_asymmetric_pattern(pattern_shape, square_size);
   size_t img_cnt;
   size_t img_with_view_cnt = collect_pattern_views(img_directory, img_cnt);
   std::cout << "Found " << img_with_view_cnt << " pattern views" << " out of " << img_cnt << std::endl;


   ceres::Problem problem;
   // one pose per image
   // TODO: read image size and set cx, cy to the center of the image

   // parameter block for camera intrinsics (standard k-matrix, no prior for distortion)
   // this block is shared between all residuals
   // double intrinsics[]; = {initial_focal_length, initial_focal_length, 1280/2, 720/2, 0.0, 0.0, 0.0, 0.0, 0.0};

   std::vector<cv::Mat> all_rvecs, all_tvecs;

   // double pattern_pose[6*all_image_points.size()];

   // init first nine parameters for camera intrinsics
   size_t param_block_size = 9 + 6*all_image_points.size(); 
   double parameters[param_block_size] = {initial_focal_length, initial_focal_length, 1280/2, 720/2, 0.0, 0.0, 0.0, 0.0, 0.0};

   for (size_t i=0; i<all_image_points.size(); i++) {
      cv::Mat rvec, tvec;
      get_initial_pose(all_image_points[i], rvec, tvec);
      all_tvecs.push_back(tvec);
      all_rvecs.push_back(rvec);

      cout << "Rot " << rvec.at<double>(0) << " " << rvec.at<double>(1) << " " << rvec.at<double>(2) << endl;
      cout << "Tra " << tvec.at<double>(0) << " " << tvec.at<double>(1) << " " << tvec.at<double>(2) << endl;


      size_t start_index = 9+i*6; // TODO: nicer version

      // TODO: get rid of magic number
      parameters[start_index+0]=rvec.at<double>(0);
      parameters[start_index+1]=rvec.at<double>(1);
      parameters[start_index+2]=rvec.at<double>(2);
      parameters[start_index+3]=tvec.at<double>(0);
      parameters[start_index+4]=tvec.at<double>(1);
      parameters[start_index+5]=tvec.at<double>(2);

      // parameter block for the pattern pose in this image
      // TODO: verify that Ceres and OpenCV use the same definition
      // TODO: keep in scope?

      problem.AddResidualBlock(
         PatternViewReprojectionError::Create(metric_pattern_points, all_image_points[i]),
         nullptr, // squared loss
         parameters + start_index,
         parameters // intrinsics at beginning of parameter array
      );

 }

   // for (size_t i=0; i<param_block_size; ++i)
   // {
   //    cout << i << "  " << parameters[i] << endl;
   // }




   cout << "Optimizing with " << all_rvecs.size() << " pattern views" << endl;

   ceres::Solver::Options options;
   options.minimizer_progress_to_stdout = true;
   ceres::Solver::Summary summary;

   ceres::Solve(options, &problem, &summary);

   std::cout << summary.FullReport() << std::endl;
   
   cout << "initial cost: " << summary.initial_cost << std::endl;

}


/**
 * @brief Compute initial pose from a single pattern view and user provided focal length
 * 
 * @param image_points extracted pattern points
 * @param rvec resulting rodriques vector
 * @param tvec resulting translation vector
 */
void RunIntrinsicCalibration::get_initial_pose(const std::vector<cv::Point2f>& image_points, cv::Mat& rvec, cv::Mat& tvec)
{
   cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
   camera_matrix.at<double>(0, 0) = initial_focal_length;
   camera_matrix.at<double>(1, 1) = initial_focal_length;
   camera_matrix.at<double>(0, 2) = 1280/2; // TODO: magic number
   camera_matrix.at<double>(1, 2) = 720/2;

   // run solvePnP without distortion and focal length provided by user
   // we could use less points to speed up computation
   bool success = cv::solvePnP(metric_pattern_points, image_points, camera_matrix, cv::Mat(), rvec, tvec);

   if (!success) {
      std::cout << "solvePnP failed" << std::endl;
      assert(false); // crash is better than unhandled error
   }

   // compute reprojection error:
   std::vector<cv::Point2f> projected_points;
   cv::projectPoints(metric_pattern_points, rvec, tvec, camera_matrix, cv::Mat(), projected_points);


   double err_sum = 0;
   for (auto tup: boost::combine(projected_points, image_points))
   {
      cv::Point2f reprojected, measured;
      boost::tie(reprojected, measured) = tup;
      err_sum += cv::norm(reprojected - measured);
   }

   double mean_error = err_sum / projected_points.size();

   if (mean_error > 20)
   {
      cerr << "Warning: initial pose has high reprojection error: " << mean_error << endl;
      cerr << "you could consider to change the initial focal length" << endl;
   }

}



size_t RunIntrinsicCalibration::collect_pattern_views(const std::filesystem::path& img_directory, size_t& num_images_in_dir)
{
   all_image_points.clear();

   num_images_in_dir = 0;
   for (const auto& entry : std::filesystem::directory_iterator(img_directory)) 
   {
      if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg"){
         num_images_in_dir++;
         std::vector<cv::Point2f> image_points;
         if (extract_pattern(entry.path(), pattern_shape, image_points)) 
         {

            cv::Mat rvec, tvec;
            get_initial_pose(image_points, rvec, tvec);

            auto z_dist = tvec.at<double>(2);

            if (z_dist < 0.1 || z_dist > 1.0) {
               cout << "for image " << entry.path() << endl;
               std::cout << "z translation " << ": " << z_dist << " is out of range, skipping" << std::endl;
               continue;
            }

            all_image_points.push_back(image_points);
         }
      }
   }

   return all_image_points.size();
}



bool RunIntrinsicCalibration::extract_pattern(const std::filesystem::path& img_path, cv::Size pattern_shape, std::vector<cv::Point2f>& image_points) 
{
   cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);
   if (img.empty()) {
      std::cerr << "Could not read image: " << img_path << std::endl;
      return false;
   }

   bool found = cv::findCirclesGrid(img, pattern_shape, image_points, cv::CALIB_CB_ASYMMETRIC_GRID);
   if (!found) {
      std::cerr << "Could not find pattern in image: " << img_path << std::endl;
      return false;
   }

   cv::Mat img_cp;
   cv::cvtColor(img, img_cp, cv::COLOR_GRAY2RGB);
   cv::drawChessboardCorners(img_cp, pattern_shape, image_points, found);
   cv::imwrite("pattern_" + img_path.filename().string(), img_cp);


   return true;
}


std::vector<cv::Point3f> RunIntrinsicCalibration::init_flat_asymmetric_pattern(cv::Size shape, double square_size)
{
   // assuming opencv assymetric pattern
   std::vector<cv::Point3f> flat_pattern_points;
   flat_pattern_points.resize(shape.width*shape.height);

   // assert(shape.width > 1);
   // assert(shape.height % 2 == 1); // enforcing asymmetric pattern
   assert(square_size > 0.0);
   assert(square_size < 0.2); // enforcing reasonable size in meters

   size_t i = 0;
   for (int y = 0; y < shape.height; ++y) {
      for (int x = 0; x < shape.width; ++x) {  
         float px = x * square_size + (y % 2) * square_size / 2;
         float py = y * square_size / 2;
         flat_pattern_points[i] = {px, py, 0.};
         // cout << "x " << x << " y " << y << " px " << px << " py " << py << endl;
         ++i;
      }
   }

   return flat_pattern_points;
}

