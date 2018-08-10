/**
 * @brief A header file with declaration for Face Detection Class
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
#include "openvino_service/outputs/base_output.h"

namespace openvino_service {

class FaceDetectionResult : public Result {
 public:
  friend class FaceDetection;
  explicit FaceDetectionResult(const cv::Rect &location);
  void decorateFrame(cv::Mat *frame, cv::Mat *camera_matrix) const override ;

 private:
  std::string label_ = "";
  float confidence_ = -1;
};

class FaceDetection : public BaseInference {
 public:
  using Result = openvino_service::FaceDetectionResult;
  explicit FaceDetection(double);
  ~FaceDetection() override;
  void loadNetwork(std::shared_ptr<Models::FaceDetectionModel>);
  bool enqueue(const cv::Mat &, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  const int getResultsLength() const override;
  const openvino_service::Result*
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
