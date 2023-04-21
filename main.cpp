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

  RunIntrinsicCalibration calib = RunIntrinsicCalibration(p, 4, 15, 0.035, 1200);

  return 0;
}

