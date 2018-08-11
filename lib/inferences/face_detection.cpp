/**
 * @brief a header file with declaration of FaceDetection class and 
 * FaceDetectionResult class
 * @file face_detection.cpp
 */
#include "openvino_service/inferences/face_detection.h"

#include "openvino_service/slog.hpp"

//FaceDetectionResult
openvino_service::FaceDetectionResult::FaceDetectionResult(
    const cv::Rect &location) : Result(location){};

void openvino_service::FaceDetectionResult::decorateFrame(
    cv::Mat *frame, cv::Mat *camera_matrix) const {
  std::ostringstream out;
  cv::Rect rect = getLocation();

  out.str("");
  if (confidence_ >= 0) {
    out << "Face Detection Confidence: "
        << std::fixed << std::setprecision(3)
        << confidence_;
  }
  cv::putText(*frame,
              out.str(),
              cv::Point2f(rect.x, rect.y - 15),
              cv::FONT_HERSHEY_COMPLEX_SMALL,
              0.8,
              cv::Scalar(0, 0, 255));
  cv::rectangle(*frame, rect, cv::Scalar(100, 100, 100), 1);
}

// FaceDetection
openvino_service::FaceDetection::FaceDetection(double show_output_thresh)
    : show_output_thresh_(show_output_thresh),
      openvino_service::BaseInference() {
};

openvino_service::FaceDetection::~FaceDetection() = default;

void openvino_service::FaceDetection::loadNetwork(
    const std::shared_ptr<Models::FaceDetectionModel> network) {
  valid_model_ = network;
  max_proposal_count_ = network->getMaxProposalCount();
  object_size_ = network->getObjectSize();
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
                                        valid_model_->getInputName())) {
    return false;
  };
  Result r(input_frame_loc);
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
  std::string output = valid_model_->getOutputName();
  const float *detections = request->GetBlob(output)->buffer().as<float *>();
  for (int i = 0; i < max_proposal_count_; i++) {
    float image_id = detections[i * object_size_ + 0];
    cv::Rect r;
    auto label_num = static_cast<int>(detections[i * object_size_ + 1]);
    std::vector<std::string> &labels = valid_model_->getLabels();
    found_result = true;
    r.x = static_cast<int>(detections[i * object_size_ + 3] *
        width_);
    r.y = static_cast<int>(detections[i * object_size_ + 4] *
        height_);
    r.width = static_cast<int>(
        detections[i * object_size_ + 5] * width_ - r.x);
    r.height = static_cast<int>(
        detections[i * object_size_ + 6] * height_ - r.y);
    Result result(r);
    result.label_ = label_num < labels.size() ? labels[label_num] :
              std::string("label #") + std::to_string(label_num);
    result.confidence_ = detections[i * object_size_ + 2];
    if (result.confidence_ <= show_output_thresh_) {
      continue;
    }

    if (image_id < 0) {
      break;
    }
    results_.emplace_back(result);
  }
  if (!found_result) results_.clear();
  return true;
};

const int openvino_service::FaceDetection::getResultsLength() const {
  return (int)results_.size();
};

const openvino_service::Result*
openvino_service::FaceDetection::getLocationResult(int idx) const {
  return &(results_[idx]);
};

const std::string
openvino_service::FaceDetection::getName() const {
  return valid_model_->getModelName();
};