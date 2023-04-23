

#include <iostream>
#include <gtest/gtest.h>
#include <ceres_calibration/ceres_intrinsic_calib.h>

using namespace std;



TEST(CalibrationTest, RunOptimization) {
  std::filesystem::path p("calib_images/synth");
  std::filesystem::path output_dir = "/tmp/output";

  // read parameters from info.yml
  cv::FileStorage info_file(p / "info.yml", cv::FileStorage::READ);
  double fx_expected = info_file["fx"];
  double fy_expected = info_file["fy"];

  double square_size = info_file["square_size"];
  cv::Size shape;
  info_file["pattern_shape"] >> shape;

  OpenCVAsymmetricCircleGridCalibration calib(output_dir);
  calib.initialize_metric_pattern(shape, square_size);
  calib.collect_pattern_views(p);

    // FIRST ROUND: read pixel positions from images, use good initial guess
    { 
        double err = calib.run_optimization(false);
        // not exactly zero because we extract the pixel positions from the images
        EXPECT_NEAR(err, 0.16, 0.01); // TODO: could also be in the info.yml

        double fx = calib.K_optimized.at<double>(0, 0);
        double fy = calib.K_optimized.at<double>(1, 1);

        // don't assert, we still want to the the rest of the test
        EXPECT_NEAR(fx, fx_expected, 0.01*fx_expected);  // allow 1% error
        EXPECT_NEAR(fx, fx_expected, 0.01*fy_expected); 
        
    }


    // SECOND ROUND:  use trivial extrinsic guess (z=0.5, centered, facing towards camera)
    {
        // remove quite well estimated initial pose
        calib.set_trivial_extrinsic_guess(0.5);

        // expect same error as before
        double err = calib.run_optimization(false);
        EXPECT_NEAR(err, 0.16, 0.01); // same error as before, problem still converges

        double fx = calib.K_optimized.at<double>(0, 0);
        double fy = calib.K_optimized.at<double>(1, 1);

        EXPECT_NEAR(fx, fx_expected, 0.01*fx_expected);  // allow 1% error
        EXPECT_NEAR(fx, fx_expected, 0.01*fy_expected); 
    }

    // THIRD ROUND:  read pixel positions from file
    {
        calib.load_test_case(p, false); // load without additional noise 
        double err = calib.run_optimization(false);
        // not exactly zero because we extract the pixel positions from the images
        EXPECT_NEAR(err, 0.0, 0.01); 

        double fx = calib.K_optimized.at<double>(0, 0);
        double fy = calib.K_optimized.at<double>(1, 1);

        EXPECT_NEAR(fx, fx_expected, 0.001);  
        EXPECT_NEAR(fx, fx_expected, 0.001); 
    }
}

TEST(SetupTest, NoMetricPoints)
{
  std::filesystem::path p("calib_images/synth");
  std::filesystem::path output_dir = "/tmp/output";

  OpenCVAsymmetricCircleGridCalibration calib(output_dir);
  // NO CALL FOR calib.initialize_metric_pattern(cv::Size(4, 15), 0.035);
  ASSERT_THROW(calib.collect_pattern_views(p), std::logic_error);
}

// TODO: real data test with images in calib_images/webcam


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}