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
 * @brief A header file with definition for EmotionDetectionModel Class
 * @file emotion_detection_model.h
 */

#ifndef OPENVINO_PIPELINE_LIB_EMOTION_DETECTION_MODEL_H
#define OPENVINO_PIPELINE_LIB_EMOTION_DETECTION_MODEL_H

#include "openvino_service/models/base_model.h"

namespace Models {

class EmotionDetectionModel : public BaseModel {
 public:
  EmotionDetectionModel(
      const std::string &, int, int, int);
  inline const std::string getInputName() { return input_; }
  inline const std::string getOutputName() { return output_; }
  const std::string getModelName() const override;

 protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override;

 private:
  std::string input_;
  std::string output_;
};

}

#endif //OPENVINO_PIPELINE_LIB_EMOTION_DETECTION_MODEL_H
