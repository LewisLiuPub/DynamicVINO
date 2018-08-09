//
// Created by chris on 18-8-9.
//

#include "openvino_service/inferences/age_gender_recognition.h"

// AgeGender Detection
openvino_service::AgeGenderDetection::AgeGenderDetection()
    : openvino_service::BaseInference() {};

openvino_service::AgeGenderDetection::~AgeGenderDetection() = default;

void openvino_service::AgeGenderDetection::loadNetwork(
    std::shared_ptr<ValidatedAgeGenderNetwork> network) {
  valid_network_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool openvino_service::AgeGenderDetection::enqueue(const cv::Mat &frame,
                                                   const cv::Rect &input_frame_loc) {
  if (getEnqueuedNum() == 0) { results_.clear(); }
  bool succeed = openvino_service::BaseInference::enqueue<float>(
      frame, input_frame_loc, 1, getResultsLength(),
      valid_network_->getInputName());
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
      genderBlob = request->GetBlob(valid_network_->getOutputGenderName());
  InferenceEngine::Blob::Ptr
      ageBlob = request->GetBlob(valid_network_->getOutputAgeName());

  for (int i = 0; i < results_.size(); ++i) {
    results_[i].age = ageBlob->buffer().as<float *>()[i] * 100;
    results_[i].male_prob = genderBlob->buffer().as<float *>()[i * 2 + 1];
  }
  return true;
};

void openvino_service::AgeGenderDetection::accepts(
    std::shared_ptr<BaseOutput> output_visitor) {
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
  return valid_network_->getNetworkName();
};