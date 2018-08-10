/**
 * @brief A header file with declaration for Age Gender Recognition Class
 * @file age_gender_recignition.h
 */
#ifndef OPENVINO_PIPELINE_LIB_AGE_GENDER_RECOGNITION_H
#define OPENVINO_PIPELINE_LIB_AGE_GENDER_RECOGNITION_H

#include <memory>

#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/engines/engine.h"
#include "openvino_service/outputs/base_output.h"
#include "openvino_service/models/age_gender_detection_model.h"

namespace openvino_service {

class AgeGenderResult : public Result {
 public:
  explicit AgeGenderResult(const cv::Rect &location);
  void decorateFrame(cv::Mat *frame, cv::Mat *camera_matrix) const override ;

  float age_ = -1;
  float male_prob_ = -1;
};

// AgeGender Detection
class AgeGenderDetection : public BaseInference {
 public:
  using Result = openvino_service::AgeGenderResult;
  explicit AgeGenderDetection();
  ~AgeGenderDetection() override;
  void loadNetwork(std::shared_ptr<Models::AgeGenderDetectionModel>);
  bool enqueue(const cv::Mat &frame, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  const int getResultsLength() const override;
  const openvino_service::Result*
  getLocationResult(int idx) const override;
  const std::string getName() const override;

 private:
  std::shared_ptr<Models::AgeGenderDetectionModel> valid_model_;
  std::vector<Result> results_;
};

}

#endif //OPENVINO_PIPELINE_LIB_AGE_GENDER_RECOGNITION_H
