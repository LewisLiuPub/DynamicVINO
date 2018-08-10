/*
 * Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
* @brief A header file with definition for ImageWindowOutput Class
* @file image_window_output.h
*/

#ifndef OPENVINO_PIPELINE_LIB_IMAGE_WINDOW_OUTPUT_H
#define OPENVINO_PIPELINE_LIB_IMAGE_WINDOW_OUTPUT_H

#include "openvino_service/outputs/base_output.h"

namespace Outputs {

class ImageWindowOutput : public BaseOutput {
 public:
  explicit ImageWindowOutput(const std::string &window_name, int);
  void prepareData(const InferenceResult::FaceDetectionResult &) override;
  void prepareData(const InferenceResult::EmotionsResult &) override;
  void prepareData(const InferenceResult::AgeGenderResult &) override;
  void prepareData(const InferenceResult::HeadPoseResult &) override;
  void feedFrame(const cv::Mat &) override;
  void handleOutput(const std::string &overall_output_text) override;

 private:
  const std::string window_name_;
  cv::Mat frame_;
  float focal_length_;
  cv::Mat camera_matrix_;
};

}
#endif //OPENVINO_PIPELINE_LIB_IMAGE_WINDOW_OUTPUT_H
