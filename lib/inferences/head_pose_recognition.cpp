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

#include "openvino_service/inferences/head_pose_recognition.h"

//Head Pose Detection
openvino_service::HeadPoseDetection::HeadPoseDetection()
    : openvino_service::BaseInference() {};

openvino_service::HeadPoseDetection::~HeadPoseDetection() = default;

void openvino_service::HeadPoseDetection::loadNetwork(
    std::shared_ptr<Models::HeadPoseDetectionModel> network) {
  valid_model_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool openvino_service::HeadPoseDetection::enqueue(const cv::Mat &frame,
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

bool openvino_service::HeadPoseDetection::submitRequest() {
  return openvino_service::BaseInference::submitRequest();
}

bool openvino_service::HeadPoseDetection::fetchResults() {
  bool can_fetch = openvino_service::BaseInference::fetchResults();
  if (!can_fetch) return false;
  auto request = getEngine()->getRequest();
  InferenceEngine::Blob::Ptr
      angle_r = request->GetBlob(valid_model_->getOutputOutputAngleR());
  InferenceEngine::Blob::Ptr
      angle_p = request->GetBlob(valid_model_->getOutputOutputAngleP());
  InferenceEngine::Blob::Ptr
      angle_y = request->GetBlob(valid_model_->getOutputOutputAngleY());

  for (int i = 0; i < getResultsLength(); ++i) {
    results_[i].angle_r = angle_r->buffer().as<float *>()[i];
    results_[i].angle_p = angle_p->buffer().as<float *>()[i];
    results_[i].angle_y = angle_y->buffer().as<float *>()[i];
  }
  return true;
};

void openvino_service::HeadPoseDetection::accepts(
    std::shared_ptr<Outputs::BaseOutput> output_visitor) {
  for (auto &result : results_) {
    output_visitor->prepareData(result);
  }
};

const int openvino_service::HeadPoseDetection::getResultsLength() const {
  return (int)results_.size();
};

const InferenceResult::Result
openvino_service::HeadPoseDetection::getLocationResult(int idx) const {
  return results_[idx];
};

const std::string openvino_service::HeadPoseDetection::getName() const {
  return valid_model_->getModelName();
};
