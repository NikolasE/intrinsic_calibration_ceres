#include <iostream>
#include <filesystem>
#include "ceres_calibration/ceres_intrinsic_calib.h"

using namespace std;
using namespace std::filesystem;

int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);

  std::filesystem::path p;
  if (argc > 1) {
    p = path(argv[1]);
    cout << "Using path: " << p << endl;
  }else{
    p = path(getenv("HOME")) / "Documents/ceres_calibration/calib_images";
    cout << "Using default path: " << p << endl;
  }

  // TODO: add interface to shape and size of pattern, or read from a config file in the directory

  // 1200: mean error: 0.52
  // 2000: 18.65
  // 800: 27.16
  // 1000: 6.34

  std::filesystem::path output_dir = p / "output";

  bool with_distortion = true;

  OpenCVAsymmetricCircleGridCalibration calib(output_dir);
  calib.initialize_metric_pattern(cv::Size(4, 15), 0.035);
  calib.collect_pattern_views(p);
  calib.run_optimization(with_distortion);




  // cout << "Running OpenCV Calibration for comparison" << endl;
  // opencv_calibrate_camera(optimize_distortion);



  return 0;
}

