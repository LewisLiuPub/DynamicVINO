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

#ifndef OPENVINO_PIPELINE_LIB_HEAD_POSE_DETECTION_MODEL_H
#define OPENVINO_PIPELINE_LIB_HEAD_POSE_DETECTION_MODEL_H

#include <openvino_service/models/base_model.h>

namespace Models {

class HeadPoseDetectionModel: public BaseModel{
 public:
  HeadPoseDetectionModel(
      const std::string &, int, int, int);
  inline const std::string getInputName() const {return input_;}
  inline const std::string getOutputOutputAngleR() const {
    return output_angle_r_;
  }
  inline const std::string getOutputOutputAngleP() const {
    return output_angle_p_;
  }
  inline const std::string getOutputOutputAngleY() const {
    return output_angle_y_;
  }
  const std::string getModelName() const override ;

 protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override ;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override ;

 private:
  std::string input_;
  std::string output_angle_r_ = "angle_r_fc";
  std::string output_angle_p_ = "angle_p_fc";
  std::string output_angle_y_ = "angle_y_fc";
};

}

#endif //OPENVINO_PIPELINE_LIB_HEAD_POSE_DETECTION_MODEL_H
