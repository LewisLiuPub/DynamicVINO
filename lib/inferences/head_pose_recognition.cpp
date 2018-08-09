//
// Created by chris on 18-8-9.
//

#include "openvino_service/inferences/head_pose_recognition.h"

//Head Pose Detection
openvino_service::HeadPoseDetection::HeadPoseDetection()
    : openvino_service::BaseInference() {};

openvino_service::HeadPoseDetection::~HeadPoseDetection() = default;

void openvino_service::HeadPoseDetection::loadNetwork(
    std::shared_ptr<ValidatedHeadPoseNetwork> network) {
  valid_network_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool openvino_service::HeadPoseDetection::enqueue(const cv::Mat &frame,
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

bool openvino_service::HeadPoseDetection::submitRequest() {
  return openvino_service::BaseInference::submitRequest();
}

bool openvino_service::HeadPoseDetection::fetchResults() {
  bool can_fetch = openvino_service::BaseInference::fetchResults();
  if (!can_fetch) return false;
  auto request = getEngine()->getRequest();
  InferenceEngine::Blob::Ptr
      angle_r = request->GetBlob(valid_network_->getOutputOutputAngleR());
  InferenceEngine::Blob::Ptr
      angle_p = request->GetBlob(valid_network_->getOutputOutputAngleP());
  InferenceEngine::Blob::Ptr
      angle_y = request->GetBlob(valid_network_->getOutputOutputAngleY());

  for (int i = 0; i < getResultsLength(); ++i) {
    results_[i].angle_r = angle_r->buffer().as<float *>()[i];
    results_[i].angle_p = angle_p->buffer().as<float *>()[i];
    results_[i].angle_y = angle_y->buffer().as<float *>()[i];
  }
  return true;
};

void openvino_service::HeadPoseDetection::accepts(
    std::shared_ptr<BaseOutput> output_visitor) {
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
  return valid_network_->getNetworkName();
};
