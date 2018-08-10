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
 * @brief A header file with definition for HeadPoseDetectionModel Class
 * @file head_pose_detection_model.h
 */
#ifndef OPENVINO_PIPELINE_LIB_BASE_OUTPUT_H
#define OPENVINO_PIPELINE_LIB_BASE_OUTPUT_H

#include "opencv2/opencv.hpp"
#include "openvino_service/data_struct.h"

namespace Outputs {
/**
 * @class BaseOutput
 * @brief This class is a base class for various output devices. It employs
 * visitor pattern to perform different operations to different inference
 * result with different output device
 */
class BaseOutput {
 public:
  BaseOutput() = default;
  virtual void prepareData(const InferenceResult::FaceDetectionResult&) = 0;
  virtual void prepareData(const InferenceResult::EmotionsResult&) = 0;
  virtual void prepareData(const InferenceResult::AgeGenderResult&) = 0;
  virtual void prepareData(const InferenceResult::HeadPoseResult&) = 0;
  virtual void handleOutput(const std::string &overall_output_text) = 0;
  virtual void feedFrame(const cv::Mat &frame) {}
};

}
#endif //OPENVINO_PIPELINE_LIB_BASE_OUTPUT_H
