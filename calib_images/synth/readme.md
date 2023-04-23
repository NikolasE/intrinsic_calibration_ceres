
Synthetic test data. 

4x15 asymmetric opencv pattern, 35mm Square_size. 

Recreate by running:

    SynthCircleProjectionsGenerator generator(cv::Size(4, 15), 0.035, cv::Size(1280, 720));
    generator.create_captures(30, "/tmp/synth");