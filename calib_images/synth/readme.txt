
Synthetic test data. 

4x15 asymmetric opencv pattern, 35mm Square_size. 

Positions are sampled so that pattern points roughly towards camera (assuming z-axis of pattern points upwards).
All information can be read from info.yml (pattern poses, pixel positions)

See IntrinsicCalibration::load_test_case on how to read this file


Recreate by running:

    SynthCircleProjectionsGenerator generator(cv::Size(4, 15), 0.035, cv::Size(1280, 720));
    generator.create_captures(30, "/tmp/synth");
