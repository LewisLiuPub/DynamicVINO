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
 * @brief A header file with definition for Emotions Recognition Class
 * @file emotions_recognition.h
 */
#ifndef OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H
#define OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H

#include <memory>

#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/engines/engine.h"
#include "openvino_service/data_struct.h"
#include "openvino_service/outputs/base_output.h"
#include "openvino_service/models/emotion_detection_model.h"

namespace openvino_service {
//Emotions Detection
class EmotionsDetection : public BaseInference {
 public:
  using Result = InferenceResult::EmotionsResult;
  explicit EmotionsDetection();
  ~EmotionsDetection() override;
  void loadNetwork(std::shared_ptr<Models::EmotionDetectionModel>);
  bool enqueue(const cv::Mat &, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<Outputs::BaseOutput> output_visitor) override;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override;
  const std::string getName() const override;

 private:
  std::shared_ptr<Models::EmotionDetectionModel> valid_model_;
  std::vector<Result> results_;
};

}

#endif //OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H
