#include "ceres_calibration/ceres_intrinsic_calib.h"
#include <iostream>
#include <math.h>

using namespace std;
namespace fs = std::filesystem;
#include <boost/range/combine.hpp>

IntrinsicCalibration::IntrinsicCalibration(const std::filesystem::path& output_directory): 
                                  output_dir(output_directory)
{
   fs::create_directories(output_directory);
}


bool IntrinsicCalibration::run_optimization(bool optimize_distortion)
{
   if (captures.empty())
   {
      cerr << "No captures found! (was collect_pattern_views called?)" << endl;
      return false;
   }

   cout << "Starting optimization " << (optimize_distortion ? "with" : "without") << " distortion" << endl;
   

   size_t param_cnt_distortion = 5; // k1, k2, p1, p2, k3

   // at least fx, fy, cx, cy, optionally k1, k2, p1, p2, k3
   size_t param_cnt_intrinsics = 4 + (optimize_distortion ? param_cnt_distortion : 0);  

   size_t param_cnt_pose = 6; // 6 for rvec and tvec 

   size_t used_capture_cnt = captures.size(); // or smaller for debugging

   
   
   double focal_length_initial = K_initial.at<double>(0, 0);
   if (focal_length_initial == 0)
   {
      cerr << "Focal length is 0, please set initial guess for K matrix" << endl;
      return false;
   }

   size_t param_block_size = param_cnt_intrinsics + param_cnt_pose*used_capture_cnt;

   // focal length and principal point are scaled by image size so that all parameters are in the same range
   double parameters[param_block_size] = {focal_length_initial/width, focal_length_initial/height, 0.5, 0.5};
   
   if (optimize_distortion)
   {
      for (int i=0; i<param_cnt_distortion; i++)
      {
         parameters[param_cnt_intrinsics+i] = 0.0; // initial guess for distortion parameters (k1, k2, p1, p2, k3
      }
   }
   

   problem.AddParameterBlock(parameters, param_cnt_intrinsics);
   
   // setParameterBounds(parameters, 0, 0.2);   // fx with 20% tolerance (already quite close)
   // setParameterBounds(parameters, 1, 0.2);   // fy 
   setParameterBounds(parameters, 2, 0.45, 0.55);   // cx 
   setParameterBounds(parameters, 3, 0.45, 0.55);   // cy 

   // use initial pose as initial guess for optimization
   for (size_t i=0; i<used_capture_cnt; i++)
   {
      const Capture& cap = captures[i];
      size_t start_index = param_cnt_intrinsics+i*param_cnt_pose; 
      problem.AddParameterBlock(parameters + start_index, param_cnt_pose);

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


      // if (optimize_distortion), 9 parameters are used for intrinsics, otherwise 4. 
      // This breaks a bit the abstraction of the cost function, but the Reprojection is currently only implemented for 4 or 9 parameters
      // TODO: check DynamicAutoDiffCostFunction if many more distortion parameters are added
      problem.AddResidualBlock(
         PatternViewReprojectionError::Create(metric_pattern_points, cap.observed_points, width, height, optimize_distortion),
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

   
   cout << "optimzied camera matrix " << endl << K_optimized << endl;
   
   if (optimize_distortion)
   {
      dist_coeffs_optimized = cv::Mat::zeros(1, 5, CV_64F);
      for (uint i=4; i<param_cnt_intrinsics; ++i) // skip fx, fy, cx, cy
      {
         dist_coeffs_optimized.at<double>(i-4) = parameters[i];
      }
      cout << "optimized distortion coeffs " << endl << dist_coeffs_optimized << endl;
   }


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
      // printf("Error changed for view %s: %.2f -> %.2f\n", cap.filename.c_str(), cap.initial_error, cap.optimized_error);

      err_sum += err;
   }

   optimized_mean_error = err_sum/used_capture_cnt;

   printf("Initial error was %.2f\n", initial_mean_error);
   printf("Optimized error is %.2f\n", optimized_mean_error);
   printf("improvement is %.2f%%\n", 100*(initial_mean_error-optimized_mean_error)/initial_mean_error);

   for (const auto&c: captures)
   {
      visualize_projections(c);
   }

   // write result to file:
   fs::path results_filename = output_dir/"calibration_result.yml";

   cv::FileStorage fs(results_filename, cv::FileStorage::WRITE);
   if (!fs.isOpened())
   {
      cout << "Could not open file " << results_filename << " for writing" << endl;
      return false;
   }

   fs << "with_distortion" << optimize_distortion;
   fs << "observation_cnt" << used_capture_cnt;
   if (optimize_distortion)
   {
      fs << "distortion_coeffs" << dist_coeffs_optimized;
   }

   fs << "initial_error" << initial_mean_error;
   fs << "optimized_error" << optimized_mean_error;
   fs << "K" << K_optimized;

   cout << "Wrote calibration results to " << results_filename << endl;


   return true;
}


/**
 * @brief Compute initial pose from a single pattern view and user provided focal length
 * 
 * @param cap
 * @return double 
 */
double IntrinsicCalibration::get_initial_pose(Capture& cap)
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

double IntrinsicCalibration::update_initial_error(Capture& cap)
{
   bool use_initial_pose = true;
   cap.initial_error = get_error(cap, use_initial_pose, K_initial);
   return cap.initial_error;
}

double IntrinsicCalibration::update_optimized_error(Capture& cap)
{
   bool use_initial_pose = false;
   cap.optimized_error = get_error(cap, use_initial_pose, K_optimized, dist_coeffs_optimized);
   return cap.optimized_error;
}

double IntrinsicCalibration::get_error(Capture& cap, bool initial, cv::Mat cam_matrix, const cv::Mat dist_coeffs)
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

double IntrinsicCalibration::collect_pattern_views(const std::filesystem::path& img_directory, size_t* num_images_in_dir)
{
   if (metric_pattern_points.empty())
   {
      cerr << "No pattern points defined. Please call initialize_metric_pattern() or similar function" << endl;
      throw std::logic_error("No pattern points defined");
   }

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


      // get pixel positions of marker features (depends on marker type)
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
   initial_mean_error = mean_error;
   return mean_error;
}


bool OpenCVAsymmetricCircleGridCalibration::visualize_detection(const Capture& cap)
{
      cv::Mat img_cp = cap.img.clone();
      cv::drawChessboardCorners(img_cp, pattern_shape, cap.observed_points, cap.pattern_visible);

      std::string debug_file_name;

      if (cap.pattern_visible)
      {
         // show first two points to make order visible
         cv::circle(img_cp, cap.observed_points[0], 10, cv::Scalar(0, 0, 255), 2);
         cv::circle(img_cp, cap.observed_points[1], 10, cv::Scalar(0, 255, 0), 2);
         debug_file_name = output_dir/("pattern__" + cap.filename+".png");
      }else
      {
         // otherwise print text to show that image was processed
         cv::putText(img_cp, "pattern not visible", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
         debug_file_name = output_dir/("no_pattern__" + cap.filename+".png");
      }

      return cv::imwrite(debug_file_name, img_cp);
}



bool OpenCVAsymmetricCircleGridCalibration::extract_pattern(Capture& cap) 
{
   // TODO: BlobDetectorParams, Thresholding preprocessing etc.
   cap.pattern_visible = cv::findCirclesGrid(cap.img, pattern_shape, cap.observed_points, cv::CALIB_CB_ASYMMETRIC_GRID);

   visualize_detection(cap);
   
   return cap.pattern_visible;
}


void OpenCVAsymmetricCircleGridCalibration::initialize_metric_pattern(cv::Size shape, double square_size_m)
{
   cout << "Searching for OpenCV Asymmetric Circle Grid calibration pattern with shape " << shape << " and square size " << square_size_m*1000 << "mm" << endl;
   if (square_size_m > 0.2)
   {
      cerr << "WARNING: square size is larger than 20cm, are you sure?" << endl;
   }

   pattern_shape = shape;
   square_size_m = square_size_m;

   metric_pattern_points.clear();
   metric_pattern_points.resize(shape.width*shape.height);

   assert(shape.width > 1);
   assert(shape.height % 2 == 1); // enforcing asymmetric pattern
   assert(square_size_m > 0.0);
   assert(square_size_m < 0.2); // enforcing reasonable size in meters

   size_t i = 0;
   for (int y = 0; y < shape.height; ++y) {
      for (int x = 0; x < shape.width; ++x) {  
         float px = x * square_size_m + (y % 2) * square_size_m / 2;
         float py = y * square_size_m / 2;
         metric_pattern_points[i] = {px, py, 0.};
         ++i;
      }
   }

}


bool IntrinsicCalibration::visualize_projections(const Capture& cap)
{
   assert(cap.projected_initial.size() == cap.projected_opt.size());

   cv::Mat img_cp = cap.img.clone();
   for (size_t i = 0; i < cap.projected_initial.size(); ++i)
   {
      cv::circle(img_cp, cap.projected_initial[i], 2, cv::Scalar(0, 0, 255), 2);
      cv::circle(img_cp, cap.projected_opt[i], 2, cv::Scalar(0, 255, 0), 2);
   }
   string filename = output_dir/("projections__" + cap.filename+".png");
   return cv::imwrite(filename, img_cp);
}

double IntrinsicCalibration::opencv_calibrate_camera(bool optimize_distortion)
{
   vector<vector<cv::Point3f>> object_points;
   vector<vector<cv::Point2f>> image_points;
   vector<cv::Mat> rvecs, tvecs;
   
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

   // convert to our error definition
   cout << "RMS error reported by cv::calibrateCamera: " << pow(rms,2) << endl;
   cout << "Optimized K matrix: " << endl << K << endl;

   if (optimize_distortion)
   {
      cout << "Optimized distortion coefficients: " << endl << dist << endl;
   }

   return rms;

   // copy results to captures
   // for (size_t i = 0; i < captures.size(); ++i)
   // {
   //    captures[i].rvec_opt = rvecs[i];
   //    captures[i].tvec_opt = tvecs[i];
   // }
   // K_optimized = K.clone();

   // double err_sum = 0;
   // for (auto& cap : captures){ err_sum += update_optimized_error(cap);}
   // double err = err_sum / captures.size();

   // cout << "Reprojection error for OpenCV Solution " << err << endl; // same as rms**2
}
