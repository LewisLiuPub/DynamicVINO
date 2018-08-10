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
 * @brief A header file with definition for Face Detection Class
 * @file face_detection.h
 */
#ifndef OPENVINO_PIPELINE_LIB_FACE_DETECTION_H
#define OPENVINO_PIPELINE_LIB_FACE_DETECTION_H

#include <memory>
#include <openvino_service/models/face_detection_model.h>

#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/engines/engine.h"
#include "openvino_service/data_struct.h"
#include "openvino_service/outputs/base_output.h"

namespace openvino_service {

class FaceDetection : public BaseInference {
 public:
  using Result = InferenceResult::FaceDetectionResult;
  explicit FaceDetection(int, int, double);
  ~FaceDetection() override;
  void loadNetwork(std::shared_ptr<Models::FaceDetectionModel>);
  bool enqueue(const cv::Mat &, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<Outputs::BaseOutput> output_visitor) override;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override;
  const std::string getName() const override;

 private:
  std::shared_ptr<Models::FaceDetectionModel> valid_model_;
  std::vector<Result> results_;
  int width_ = 0;
  int height_ = 0;
  int max_proposal_count_;
  int object_size_;
  double show_output_thresh_ = 0;
};

}
#endif //OPENVINO_PIPELINE_LIB_FACE_DETECTION_H
