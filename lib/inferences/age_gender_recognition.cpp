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

#include "openvino_service/inferences/age_gender_recognition.h"

// AgeGender Detection
openvino_service::AgeGenderDetection::AgeGenderDetection()
    : openvino_service::BaseInference() {};

openvino_service::AgeGenderDetection::~AgeGenderDetection() = default;

void openvino_service::AgeGenderDetection::loadNetwork(
    std::shared_ptr<Models::AgeGenderDetectionModel> network) {
  valid_model_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool openvino_service::AgeGenderDetection::enqueue(const cv::Mat &frame,
                                                   const cv::Rect &input_frame_loc) {
  if (getEnqueuedNum() == 0) { results_.clear(); }
  bool succeed = openvino_service::BaseInference::enqueue<float>(
      frame, input_frame_loc, 1, getResultsLength(),
      valid_model_->getInputName());
  if (!succeed ) return false;
  Result r;
  r.location = input_frame_loc;
  results_.emplace_back(r);
  return true;
}

bool openvino_service::AgeGenderDetection::submitRequest() {
  return openvino_service::BaseInference::submitRequest();
}

bool openvino_service::AgeGenderDetection::fetchResults() {
  bool can_fetch = openvino_service::BaseInference::fetchResults();
  if (!can_fetch) return false;
  auto request = getEngine()->getRequest();
  InferenceEngine::Blob::Ptr
      genderBlob = request->GetBlob(valid_model_->getOutputGenderName());
  InferenceEngine::Blob::Ptr
      ageBlob = request->GetBlob(valid_model_->getOutputAgeName());

  for (int i = 0; i < results_.size(); ++i) {
    results_[i].age = ageBlob->buffer().as<float *>()[i] * 100;
    results_[i].male_prob = genderBlob->buffer().as<float *>()[i * 2 + 1];
  }
  return true;
};

void openvino_service::AgeGenderDetection::accepts(
    std::shared_ptr<Outputs::BaseOutput> output_visitor) {
  for (auto &result : results_) {
    output_visitor->prepareData(result);
  }
};

const int openvino_service::AgeGenderDetection::getResultsLength() const {
  return (int)results_.size();
};

const InferenceResult::Result
openvino_service::AgeGenderDetection::getLocationResult(int idx) const {
  return results_[idx];
};

const std::string openvino_service::AgeGenderDetection::getName() const {
  return valid_model_->getModelName();
};