/**
 * @brief A header file with declaration for FaceDetection Class
 * @file head_pose_recognition.h
 */
#ifndef OPENVINO_PIPELINE_LIB_HEAD_POSE_RECOGNITION_H
#define OPENVINO_PIPELINE_LIB_HEAD_POSE_RECOGNITION_H

#include <memory>

#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/engines/engine.h"
#include "openvino_service/outputs/base_output.h"
#include "openvino_service/models/head_pose_detection_model.h"

namespace openvino_service {

//HeadPoseResult
class HeadPoseResult : public Result {
 public:
  friend class HeadPoseDetection;
  explicit HeadPoseResult(const cv::Rect &location);
  void decorateFrame(cv::Mat *frame, cv::Mat *camera_matrix) const override ;

 private:
  float angle_y_ = -1;
  float angle_p_ = -1;
  float angle_r_ = -1;
};

// Head Pose Detection
class HeadPoseDetection : public BaseInference {
 public:
  using Result = openvino_service::HeadPoseResult;
  explicit HeadPoseDetection();
  ~HeadPoseDetection() override;
  void loadNetwork(std::shared_ptr<Models::HeadPoseDetectionModel>);
  bool enqueue(const cv::Mat &frame, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  const int getResultsLength() const override;
  const openvino_service::Result*
  getLocationResult(int idx) const override;
  const std::string getName() const override;

 private:
  std::shared_ptr<Models::HeadPoseDetectionModel> valid_model_;
  std::vector<Result> results_;
};

}
#endif //OPENVINO_PIPELINE_LIB_HEAD_POSE_RECOGNITION_H
