//
// Created by chris on 18-8-9.
//

#include "openvino_service/inferences/face_detection.h"

#include "openvino_service/slog.hpp"

// FaceDetection
openvino_service::FaceDetection::FaceDetection(
    int max_proposal_count, int object_size, double show_output_thresh)
    : max_proposal_count_(max_proposal_count),
      object_size_(object_size),
      show_output_thresh_(show_output_thresh), openvino_service::BaseInference() {
};

openvino_service::FaceDetection::~FaceDetection() = default;

void openvino_service::FaceDetection::loadNetwork(
    const std::shared_ptr<ValidatedFaceDetectionNetwork> network) {
  valid_network_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool
openvino_service::FaceDetection::enqueue(
    const cv::Mat &frame, const cv::Rect &input_frame_loc) {
  if (width_ == 0 && height_ == 0) {
    width_ = frame.cols;
    height_ = frame.rows;
  }
  if (!openvino_service::BaseInference::enqueue<u_int8_t>(frame, input_frame_loc, 1, 0,
                                        valid_network_->getInputName())) {
    return false;
  };
  Result r;
  r.location = input_frame_loc;
  results_.clear();
  results_.emplace_back(r);
  return true;
};

bool openvino_service::FaceDetection::submitRequest() {
  return openvino_service::BaseInference::submitRequest();
};

bool openvino_service::FaceDetection::fetchResults() {
  bool can_fetch = openvino_service::BaseInference::fetchResults();
  if (!can_fetch) return false;
  bool found_result = false;
  results_.clear();
  InferenceEngine::InferRequest::Ptr request = getEngine()->getRequest();
  std::string output = valid_network_->getOutputName();
  const float *detections = request->GetBlob(output)->buffer().as<float *>();
  for (int i = 0; i < max_proposal_count_; i++) {
    float image_id = detections[i * object_size_ + 0];
    Result r;
    auto label_num = static_cast<int>(detections[i * object_size_ + 1]);
    std::vector<std::string> &labels = valid_network_->getLabels();
    r.label = label_num < labels.size() ? labels[label_num] :
              std::string("label #") + std::to_string(label_num);
    r.confidence = detections[i * object_size_ + 2];
    if (r.confidence <= show_output_thresh_) {
      continue;
    }
    found_result = true;
    r.location.x = static_cast<int>(detections[i * object_size_ + 3] *
        width_);
    r.location.y = static_cast<int>(detections[i * object_size_ + 4] *
        height_);
    r.location.width = static_cast<int>(
        detections[i * object_size_ + 5] * width_ - r.location.x);
    r.location.height = static_cast<int>(
        detections[i * object_size_ + 6] * height_ - r.location.y);

    if (image_id < 0) {
      break;
    }
    results_.emplace_back(r);
  }
  if (!found_result) results_.clear();
  return true;
};

void openvino_service::FaceDetection::accepts(
    std::shared_ptr<BaseOutput> output_visitor) {
  for (auto &result : results_) {
    output_visitor->prepareData(result);
  }
}

const int openvino_service::FaceDetection::getResultsLength() const {
  return (int)results_.size();
};

const InferenceResult::Result
openvino_service::FaceDetection::getLocationResult(int idx) const {
  return results_[idx];
};

const std::string
openvino_service::FaceDetection::getName() const {
  return valid_network_->getNetworkName();
};