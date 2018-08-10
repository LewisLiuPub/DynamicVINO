#include "openvino_service/inferences/age_gender_recognition.h"

//AgeGenderResult
openvino_service::AgeGenderResult::AgeGenderResult(
    const cv::Rect &location) : Result(location){};

void openvino_service::AgeGenderResult::decorateFrame(
    cv::Mat *frame, cv::Mat *camera_matrix) const {
  std::ostringstream out;
  cv::Rect rect = getLocation();

  out.str("");
  out << "Age: " << age_ << "," <<
      "Gender: " << ((male_prob_ > 0.5)?"Male":"Female");
  cv::putText(*frame,
              out.str(),
              cv::Point2f(rect.x, rect.y-5),
              cv::FONT_HERSHEY_COMPLEX_SMALL,
              0.8,
              cv::Scalar(0, 255, 0));
  cv::rectangle(*frame, rect, cv::Scalar(100, 100, 100), 1);
}

// AgeGender Detection
openvino_service::AgeGenderDetection::AgeGenderDetection()
    : openvino_service::BaseInference() {};

openvino_service::AgeGenderDetection::~AgeGenderDetection() = default;

void openvino_service::AgeGenderDetection::loadNetwork(
    std::shared_ptr<Models::AgeGenderDetectionModel> network) {
  valid_model_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool openvino_service::AgeGenderDetection::enqueue(
    const cv::Mat &frame,
    const cv::Rect &input_frame_loc) {
  if (getEnqueuedNum() == 0) { results_.clear(); }
  bool succeed = openvino_service::BaseInference::enqueue<float>(
      frame, input_frame_loc, 1, getResultsLength(),
      valid_model_->getInputName());
  if (!succeed ) return false;
  Result r(input_frame_loc);
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
    results_[i].age_ = ageBlob->buffer().as<float *>()[i] * 100;
    results_[i].male_prob_ = genderBlob->buffer().as<float *>()[i * 2 + 1];
  }
  return true;
};

const int openvino_service::AgeGenderDetection::getResultsLength() const {
  return (int)results_.size();
};

const openvino_service::Result*
openvino_service::AgeGenderDetection::getLocationResult(int idx) const {
  return &(results_[idx]);
};

const std::string openvino_service::AgeGenderDetection::getName() const {
  return valid_model_->getModelName();
};