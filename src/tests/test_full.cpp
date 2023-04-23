

#include <gtest/gtest.h>
#include <ceres_calibration/ceres_intrinsic_calib.h>

TEST(CalibrationTest, RunOptimization) {
  std::filesystem::path p("/tmp/synth");
  std::filesystem::path output_dir = p / "output";

  OpenCVAsymmetricCircleGridCalibration calib(output_dir);
  calib.initialize_metric_pattern(cv::Size(4, 15), 0.035);
  calib.collect_pattern_views(p);

  calib.set_trivial_extrinsic_guess();

  double err = calib.run_optimization(false);
  EXPECT_NEAR(err, 0.16, 1e-6); // check that the error is close to zero
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}