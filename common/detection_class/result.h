//
// Created by chris on 18-8-2.
//

#ifndef SAMPLES_INFERENCERESULT_H
#define SAMPLES_INFERENCERESULT_H

#include "opencv2/opencv.hpp"

namespace InferenceResult {
struct Result {
  cv::Rect location;
};
struct FaceDetectionResult : Result {
  std::string label = "";
  float confidence = -1;
};
struct EmotionsResult : Result {
  std::string label = "";
  float confidence = -1;
};
struct AgeGenderResult : Result {
  float age = -1;
  float male_prob = -1;
};
struct HeadPoseResult : Result {
  float angle_y = -1;
  float angle_p = -1;
  float angle_r = -1;
};
}

#endif //SAMPLES_INFERENCERESULT_H
