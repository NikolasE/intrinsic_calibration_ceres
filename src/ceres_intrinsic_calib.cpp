#include "ceres_calibration/ceres_intrinsic_calib.h"
#include <iostream>
#include <math.h>

using namespace std;
namespace fs = std::filesystem;
#include <boost/range/combine.hpp>

RunIntrinsicCalibration::RunIntrinsicCalibration(const std::filesystem::path& img_directory,
                                  uint pattern_width, uint pattern_height,
                                  double square_size): 
                                  pattern_shape(pattern_width, pattern_height),
                                  square_size(square_size)
{

   printf("Starting optimization for %ix%i pattern with size of %.fmm, looking for images in %s\n",
          pattern_width, pattern_height, square_size*1000, img_directory.c_str());


   // create debug-directory, include focal length in name
   debug_directory = img_directory / "debug";
   fs::create_directories(debug_directory);


   metric_pattern_points = RunIntrinsicCalibration::init_flat_asymmetric_pattern(pattern_shape, square_size);
   double initial_mean_error = collect_pattern_views(img_directory); // also finds good initial estimate for focal length


   std::vector<cv::Mat> all_rvecs, all_tvecs;

   size_t param_cnt_intrinsics = 4; // 9 with distortion

   size_t used_capture_cnt = captures.size(); // or smaller for debugging


   // init first parameters for camera intrinsics
   size_t param_block_size = param_cnt_intrinsics + 6*used_capture_cnt;
   
   double focal_length_initial = K_initial.at<double>(0, 0);


   // focal length and principal point are scaled by image size so that all parameters are in the same range
   double parameters[param_block_size] = {focal_length_initial/width, focal_length_initial/height, 0.5, 0.5}; 
   // double parameters[param_block_size] = {0.8, 1.6, 0.5, 0.5}; 
   

   problem.AddParameterBlock(parameters, param_cnt_intrinsics);
   
   // setParameterBounds(parameters, 0, 0.2);   // fx with 20% tolerance (already quite close)
   // setParameterBounds(parameters, 1, 0.2);   // fy 
   setParameterBounds(parameters, 2, 0.45, 0.55);   // cx 
   setParameterBounds(parameters, 3, 0.45, 0.55);   // cy 

   // use initial pose as initial guess for optimization
   for (size_t i=0; i<used_capture_cnt; i++)
   {
      const Capture& cap = captures[i];
      size_t start_index = param_cnt_intrinsics+i*6; 
      problem.AddParameterBlock(parameters + start_index, 6);

      for (int i: {0,1,2})
      {
         parameters[start_index+i] = cap.rvec_initial.at<double>(i);
         parameters[start_index+i+3] = cap.tvec_initial.at<double>(i);

         setParameterBounds(parameters + start_index, i, -M_PI, M_PI); // if required, set after AddResidualBlock!
         setParameterBounds(parameters + start_index, i+3, -1, 1);
      }
      setParameterBounds(parameters + start_index, 5, 0, 1); // z always positive

      {
         // OPTIONAL DEBUG CHECK:
         // directly call costfunction to compare with our initial result
         // this is important to check that we use the same Rodrigues definition and projection model like OpenCV
         // to compare results and helps as a sanity check. 
   
         //    PatternViewReprojectionError reproError(metric_pattern_points, cap.observed_points, width, height);
         //    double error = 0;
         //    reproError(parameters + start_index, parameters, &error);
         //    if (fabs(error - cap.initial_error) > 1e-2)
         //       throw std::runtime_error("Initial error does not match: " + std::to_string(error) + " vs " + std::to_string(cap.initial_error));
      }  


      // now add residual block
      problem.AddResidualBlock(
         PatternViewReprojectionError::CreateNoDistortion(metric_pattern_points, cap.observed_points, width, height),
         nullptr, // squared loss
         parameters + start_index,
         parameters // intrinsics at beginning of parameter array
      );

   }

   cout << "Optimizing with " << used_capture_cnt << " pattern views" << endl;

   // copy parameters-array to compare later
   double parameters_copy[param_block_size];
   memcpy(parameters_copy, parameters, sizeof(parameters));

   ceres::Solver::Options options;
   options.minimizer_progress_to_stdout = true;
   options.max_num_iterations = 500;
   // options.linear_solver_type = ceres::DENSE_SCHUR;
   options.linear_solver_type = ceres::DENSE_QR;
   
   ceres::Solver::Summary summary;

   ceres::Solve(options, &problem, &summary);

   std::cout << summary.FullReport() << std::endl;

   // for (size_t i=0; i<param_block_size; ++i)
   // {
   //    cout << i << "  " << parameters_copy[i] << " -> " << parameters[i] << endl;
   // }
   
   // recompute reprojection error to compare with result of Ceres (should prove that we use the same Rodrigues definition)
   K_optimized = cv::Mat::eye(3, 3, CV_64F);
   K_optimized.at<double>(0, 0) = parameters[0]*width;
   K_optimized.at<double>(1, 1) = parameters[1]*height;
   K_optimized.at<double>(0, 2) = parameters[2]*width;
   K_optimized.at<double>(1, 2) = parameters[3]*height;

   // in future:
   // dist_coeffs_optimized = parameters[....]

   cout << "new camera matrix " << endl << K_optimized << endl;

   double err_sum = 0;

   for (size_t i=0; i<used_capture_cnt; ++i)
   {
      size_t start_index = param_cnt_intrinsics+i*6;
      Capture& cap = captures[i];

      for (int j: {0,1,2})
      {
         cap.rvec_opt.at<double>(j) = parameters[start_index+j];
         cap.tvec_opt.at<double>(j) = parameters[start_index+j+3];
      }

      double err = update_optimized_error(cap); // using K_optimized
      printf("Error changed for view %s: %.2f -> %.2f\n", cap.filename.c_str(), cap.initial_error, cap.optimized_error);

      err_sum += err;
   }

   double mean_err = err_sum/used_capture_cnt;

   printf("Initial error was %.2f\n", initial_mean_error);
   printf("Optimized error is %.2f\n", mean_err);
   printf("improvement is %.2f%%\n", 100*(initial_mean_error-mean_err)/initial_mean_error);

   for (const auto&c: captures)
   {
      visualize_projections(c);
   }

   cout << "Running OpenCV Calibration for comparison" << endl;
   opencv_calibrate_camera();


   // TODO: write results to file
}


/**
 * @brief Compute initial pose from a single pattern view and user provided focal length
 * 
 * @param cap
 * @return double 
 */
double RunIntrinsicCalibration::get_initial_pose(Capture& cap)
{
   // make sure that init_flat_asymmetric_pattern (or similar function) was called before
   assert(cap.observed_points.size() == metric_pattern_points.size());

   // run solvePnP without distortion and focal length provided by user
   // we could use less points to speed up computation
   bool success = cv::solvePnP(metric_pattern_points, cap.observed_points, K_initial, cv::Mat(), cap.rvec_initial, cap.tvec_initial);

   if (!success) {
      cout << "solvePnP failed" << endl;
      assert(false); // crash is better than unhandled error. We'd need to remove the Capture in this case
   }

   double z = cap.tvec_initial.at<double>(2);
   if (z<0 || z > 2)
   {
      cerr << "estimated z value for pattern pose in image " << cap.filename << " is " << z << endl;
      cerr << "this looks wrong, adjust initial focal length or debug further" << endl;
      cerr << "This is really unexpected. Please report this issue to the calibration team" << endl;
      exit(23);
   }

   return update_initial_error(cap);
}

double RunIntrinsicCalibration::update_initial_error(Capture& cap)
{
   bool use_initial_pose = true;
   cap.initial_error = get_error(cap, use_initial_pose, K_initial);
   return cap.initial_error;
}

double RunIntrinsicCalibration::update_optimized_error(Capture& cap)
{
   bool use_initial_pose = false;
   cap.optimized_error = get_error(cap, use_initial_pose, K_optimized, dist_coeffs_optimized);
   return cap.optimized_error;
}

double RunIntrinsicCalibration::get_error(Capture& cap, bool initial, cv::Mat cam_matrix, const cv::Mat dist_coeffs)
{
   assert(metric_pattern_points.size() == cap.observed_points.size());
   assert(metric_pattern_points.size() > 0);

   cv::Mat& rvec = initial ? cap.rvec_initial : cap.rvec_opt;
   cv::Mat& tvec = initial ? cap.tvec_initial : cap.tvec_opt;

   assert(!rvec.empty());

   if (cam_matrix.empty())
   {
      cam_matrix = K_initial;
   }

   std::vector<cv::Point2f>& projected_points = initial ? cap.projected_initial : cap.projected_opt;
   cv::projectPoints(metric_pattern_points, rvec, tvec, cam_matrix, dist_coeffs, projected_points);

   double err_sum = 0;
   cv::Point2f reprojected, observed;
   for (auto tup: boost::combine(projected_points, cap.observed_points))
   {
      boost::tie(reprojected, observed) = tup;
      err_sum += pow(cv::norm(reprojected - observed),2); // TODO: decide if we want square or not
   }

   double mean_error = err_sum / projected_points.size();
   return mean_error;
}

double RunIntrinsicCalibration::collect_pattern_views(const std::filesystem::path& img_directory, size_t* num_images_in_dir)
{
   captures.clear();
   double err_sum = 0;

   size_t img_cnt = 0;
   // TODO: maybe support some regex
   for (const auto& entry : std::filesystem::directory_iterator(img_directory)) 
   {
      if (!(entry.path().extension() == ".png" || entry.path().extension() == ".jpg"))
      {
         continue;
      }

      img_cnt++;

      Capture cap(entry.path());

      // read image
      cap.img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
      if (cap.img.empty()) 
      {
         std::cerr << "Could not read image: " << entry.path() << std::endl;
         continue;
      }


      // get pixel positions of circles
      if (!extract_pattern(cap))
      {
         // cerr << "no pattern found in image " << cap.filename << endl;
         continue;
      }

      // check if this is the first image, in this case remember size and compare other images against it
      if (captures.empty())
      {
         width = cap.img.cols;
         height = cap.img.rows;
         cout << "Found first image with size: " << width << " x " << height << endl;

         K_initial = cv::Mat::eye(3, 3, CV_64F);
         K_initial.at<double>(0, 2) = width/2; 
         K_initial.at<double>(1, 2) = height/2;


         cout << "sampling values for focal length:" << endl;

         // search for good initial focal length:
         double f = 400;
         double best_err = -1;
         double best_f = -1;
         // TODO: check if this is a good range
         for (int i=0; i<20; i++) // search from 400 to 400+20*100
         {
            f += 100;
            K_initial.at<double>(0, 0) = f;
            K_initial.at<double>(1, 1) = f;
            double err = get_initial_pose(cap);
            printf("error for f=%.2f: %.2f\n", f, err);

            if (err < best_err || best_err < 0)
            {
               best_err = err;
               best_f = f;
            }

            // error has single minimum, so we can stop as soon as the error is increasing again
            if (err > best_err)
            {
               K_initial.at<double>(0, 0) = best_f;
               K_initial.at<double>(1, 1) = best_f;
               cout << "Using f=" << best_f << " for initial pose estimation" << endl;
               break;
            }
         }
      }else
      {
         if (width != cap.img.cols || height != cap.img.rows)
         {
            std::cerr << "Image " << entry.path() << " has different size than previous images" << std::endl;
            // or terminate completely?
            continue;
         }
      }



      // compute initial position and its error
      double err = get_initial_pose(cap);

      // we now have a capture with one visible pattern and an initial pose
      captures.push_back(cap);


      double z = cap.tvec_initial.at<double>(2);

      printf("initial error: %zu %.2f with z-distance of %.2f\n", captures.size()-1, err, z);
      err_sum += err;      
   }

   if (captures.empty())
   {
      cerr << "could not find any images with patterns in " << img_directory << endl;
      exit(1);
   }

   double mean_error = err_sum / captures.size();

   cout << "Found " << captures.size() << " images with patterns in " << img_cnt << " images" << endl;

   // should be done within the loop, but this looks fancy
   double max_error = std::max_element(captures.begin(), captures.end(), [](const Capture& a, const Capture& b) { return a.initial_error < b.initial_error; })->initial_error;

   printf("mean error: %.2f, max error: %.2f\n", mean_error, max_error);

   std::sort(captures.begin(), captures.end(), [](const Capture& a, const Capture& b) { return a.filename < b.filename; });


   if (num_images_in_dir){ *num_images_in_dir = img_cnt;}
   return mean_error;
}


bool RunIntrinsicCalibration::extract_pattern(Capture& cap) 
{
   // TODO: BlobDetectorParams, Thresholding preporcessing etc.
   cap.pattern_visible = cv::findCirclesGrid(cap.img, pattern_shape, cap.observed_points, cv::CALIB_CB_ASYMMETRIC_GRID);

   { // create debug image
      cv::Mat img_cp = cap.img.clone();
      cv::drawChessboardCorners(img_cp, pattern_shape, cap.observed_points, cap.pattern_visible);

      std::string debug_file_name;

      if (cap.pattern_visible)
      {
         // show first two points to make order visible
         cv::circle(img_cp, cap.observed_points[0], 10, cv::Scalar(0, 0, 255), 2);
         cv::circle(img_cp, cap.observed_points[1], 10, cv::Scalar(0, 255, 0), 2);
         debug_file_name = debug_directory/("pattern__" + cap.filename+".png");
      }else
      {
         // otherwise print text to show that image was processed
         cv::putText(img_cp, "pattern not visible", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
         debug_file_name = debug_directory/("no_pattern__" + cap.filename+".png");
      }

      cv::imwrite(debug_file_name, img_cp);
   }
   
   return cap.pattern_visible;
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


void RunIntrinsicCalibration::visualize_projections(const Capture& cap)
{
   assert(cap.projected_initial.size() == cap.projected_opt.size());


   cv::Mat img_cp = cap.img.clone();
   for (size_t i = 0; i < cap.projected_initial.size(); ++i)
   {
      cv::circle(img_cp, cap.projected_initial[i], 2, cv::Scalar(0, 0, 255), 2);
      cv::circle(img_cp, cap.projected_opt[i], 2, cv::Scalar(0, 255, 0), 2);
      // cv::line(img_cp, cap.projected_initial[i], cap.projected_opt[i], cv::Scalar(255, 0, 0), 2);
   }
   string filename = debug_directory/("projections__" + cap.filename+".png");
   cv::imwrite(filename, img_cp);
}

void RunIntrinsicCalibration::opencv_calibrate_camera(bool optimize_distortion)
{
   vector<vector<cv::Point3f>> object_points;
   vector<vector<cv::Point2f>> image_points;
   vector<cv::Mat> rvecs, tvecs;
   // vector<double> per_view_errors;
   
   for (size_t i = 0; i < captures.size(); ++i)
   {
      object_points.push_back(metric_pattern_points); // same for all
      image_points.push_back(captures[i].observed_points);
      rvecs.push_back(captures[i].rvec_initial); // CALIB_USE_EXTRINSIC_GUESS
      tvecs.push_back(captures[i].tvec_initial);
   }

   cv::Size image_size = cv::Size(width, height);


   cv::Mat dist;
   int flags = cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_USE_EXTRINSIC_GUESS;
   if (optimize_distortion)
   {
      dist = cv::Mat::zeros(5, 1, CV_64F); 
   }else{
      flags |= cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_ZERO_TANGENT_DIST;    
   }

   cv::Mat K = K_initial.clone();

   double rms = cv::calibrateCamera(object_points, image_points, image_size, K, dist, rvecs, tvecs, flags);

   cout << "RMS error reported by cv::calibrateCamera: " << rms << endl;
   cout << "Optimized K matrix: " << endl << K << endl;


   // copy results to captures
   // for (size_t i = 0; i < captures.size(); ++i)
   // {
   //    captures[i].rvec_opt = rvecs[i];
   //    captures[i].tvec_opt = tvecs[i];
   // }



   // TODO: copy rvec, tvec to capture, redo evaluation
}
