/**
 * @brief A header file with declaration for EmotionsDetection Class
 * @file emotions_recognition.h
 */
#ifndef OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H
#define OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H

#include <memory>

#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/engines/engine.h"
#include "openvino_service/outputs/base_output.h"
#include "openvino_service/models/emotion_detection_model.h"

namespace openvino_service {

class EmotionsResult : public Result {
 public:
  friend class EmotionsDetection;
  explicit EmotionsResult(const cv::Rect &location);
  void decorateFrame(cv::Mat *frame, cv::Mat *camera_matrix) const override ;

 private:
  std::string label_ = "";
  float confidence_ = -1;
};

//Emotions Detection
class EmotionsDetection : public BaseInference {
 public:
  using Result = openvino_service::EmotionsResult;
  explicit EmotionsDetection();
  ~EmotionsDetection() override;
  void loadNetwork(std::shared_ptr<Models::EmotionDetectionModel>);
  bool enqueue(const cv::Mat &, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  const int getResultsLength() const override;
  const openvino_service::Result*
  getLocationResult(int idx) const override;
  const std::string getName() const override;

 private:
  std::shared_ptr<Models::EmotionDetectionModel> valid_model_;
  std::vector<Result> results_;
};

}

#endif //OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H
