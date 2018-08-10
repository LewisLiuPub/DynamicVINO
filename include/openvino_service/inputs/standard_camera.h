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
 * @brief A header file with definition for Standard Camera
 * @file standard_camera.h
 */

#ifndef OPENVINO_PIPELINE_LIB_STANDARD_CAMERA_H
#define OPENVINO_PIPELINE_LIB_STANDARD_CAMERA_H

#include "openvino_service/inputs/base_input.h"

#include <opencv2/opencv.hpp>

namespace Input {
class StandardCamera : public BaseInputDevice {
 public:
  bool initialize() override;
  bool initialize(int t) override;
  bool initialize(size_t width, size_t height) override;
  bool read(cv::Mat *frame) override;
  void config() override;

 private:
  cv::VideoCapture cap;
};
}
#endif //OPENVINO_PIPELINE_LIB_STANDARD_CAMERA_H
