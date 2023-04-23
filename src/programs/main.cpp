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
    // p = path(getenv("HOME")) / "Documents/ceres_calibration/calib_images";
    p = path("/tmp/synth");

    cout << "Using default path: " << p << endl;
  }

  std::filesystem::path output_dir = p / "output";

  bool with_distortion = false;

  OpenCVAsymmetricCircleGridCalibration calib(output_dir);
  calib.initialize_metric_pattern(cv::Size(4, 15), 0.035);
  calib.collect_pattern_views(p);

  calib.set_trivial_extrinsic_guess();

  double err = calib.run_optimization(with_distortion);
  cout << "Error: " << err << endl; // should be zero for perfect test data

  // bool add_noise = true;
  // calib.load_test_case(p, add_noise);
  // double err = calib.run_optimization(with_distortion);
  // cout << "Error: " << err << endl; // should be zero for perfect test data

  // cout << "Running OpenCV Calibration for comparison" << endl;
  // opencv_calibrate_camera(optimize_distortion);

  return 0;
}

