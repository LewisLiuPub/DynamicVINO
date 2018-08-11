/**
 * @brief a header file with declaration of HeadPoseDetection class and 
 * HeadPoseResult class
 * @file head_pose_recognition.cpp
 */
#include "openvino_service/inferences/head_pose_recognition.h"

//HeadPoseResult
openvino_service::HeadPoseResult::HeadPoseResult(const cv::Rect &location) :
Result(location){};

void openvino_service::HeadPoseResult::decorateFrame(
    cv::Mat *frame, cv::Mat *camera_matrix) const {
  int scale  = 50;
  int cx = frame->cols / 2;
  int cy = frame->rows / 2;
  double yaw = angle_y_ * CV_PI / 180.0;
  double pitch = angle_p_ * CV_PI / 180.0;
  double roll = angle_r_ * CV_PI / 180.0;
  cv::Rect rect = getLocation();
  cv::Point3f cpoint(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
  cv::Matx33f Rx(1, 0, 0,
                 0, cos(pitch), -sin(pitch),
                 0, sin(pitch), cos(pitch));
  cv::Matx33f Ry(cos(yaw), 0, -sin(yaw),
                 0, 1, 0,
                 sin(yaw), 0, cos(yaw));
  cv::Matx33f Rz(cos(roll), -sin(roll), 0,
                 sin(roll), cos(roll), 0,
                 0, 0, 1);

  auto r = cv::Mat(Rz * Ry * Rx);

  cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F),
      zAxis1(3, 1, CV_32F);

  xAxis.at<float>(0) = 1 * scale;
  xAxis.at<float>(1) = 0;
  xAxis.at<float>(2) = 0;

  yAxis.at<float>(0) = 0;
  yAxis.at<float>(1) = -1 * scale;
  yAxis.at<float>(2) = 0;

  zAxis.at<float>(0) = 0;
  zAxis.at<float>(1) = 0;
  zAxis.at<float>(2) = -1 * scale;

  zAxis1.at<float>(0) = 0;
  zAxis1.at<float>(1) = 0;
  zAxis1.at<float>(2) = 1 * scale;

  cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
  o.at<float>(2) = camera_matrix->at<float>(0);

  xAxis = r * xAxis + o;
  yAxis = r * yAxis + o;
  zAxis = r * zAxis + o;
  zAxis1 = r * zAxis1 + o;

  cv::Point p1, p2;

  p2.x = static_cast<int>(
      (xAxis.at<float>(0) / xAxis.at<float>(2) * camera_matrix->at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (xAxis.at<float>(1) / xAxis.at<float>(2) * camera_matrix->at<float>(4))
          + cpoint.y);
  cv::line(*frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

  p2.x = static_cast<int>(
      (yAxis.at<float>(0) / yAxis.at<float>(2) * camera_matrix->at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (yAxis.at<float>(1) / yAxis.at<float>(2) * camera_matrix->at<float>(4))
          + cpoint.y);
  cv::line(*frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

  p1.x = static_cast<int>(
      (zAxis1.at<float>(0) / zAxis1.at<float>(2) * camera_matrix->at<float>(0))
          + cpoint.x);
  p1.y = static_cast<int>(
      (zAxis1.at<float>(1) / zAxis1.at<float>(2) * camera_matrix->at<float>(4))
          + cpoint.y);

  p2.x = static_cast<int>(
      (zAxis.at<float>(0) / zAxis.at<float>(2) * camera_matrix->at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (zAxis.at<float>(1) / zAxis.at<float>(2) * camera_matrix->at<float>(4))
          + cpoint.y);
  cv::line(*frame, p1, p2, cv::Scalar(255, 0, 0), 2);
  cv::circle(*frame, p2, 3, cv::Scalar(255, 0, 0), 2);
};

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
  Result r(input_frame_loc);
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
    results_[i].angle_r_ = angle_r->buffer().as<float *>()[i];
    results_[i].angle_p_ = angle_p->buffer().as<float *>()[i];
    results_[i].angle_y_ = angle_y->buffer().as<float *>()[i];
  }
  return true;
};

const int openvino_service::HeadPoseDetection::getResultsLength() const {
  return (int)results_.size();
};

const openvino_service::Result*
openvino_service::HeadPoseDetection::getLocationResult(int idx) const {
  return &(results_[idx]);
};

const std::string openvino_service::HeadPoseDetection::getName() const {
  return valid_model_->getModelName();
};
