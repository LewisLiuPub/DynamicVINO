//
// Created by chris on 18-8-9.
//

#ifndef OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H
#define OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H

#include <memory>

#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"

#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/engines/engine.h"
#include "openvino_service/data_struct.h"
#include "openvino_service/outputs/output.h"

namespace openvino_service {
//Emotions Detection
class EmotionsDetection : public BaseInference {
 public:
  using Result = InferenceResult::EmotionsResult;
  explicit EmotionsDetection();
  ~EmotionsDetection() override;
  void loadNetwork(std::shared_ptr<ValidatedEmotionsClassificationNetwork>);
  bool enqueue(const cv::Mat &, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<BaseOutput> output_visitor) override;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override;
  const std::string getName() const override;

 private:
  std::shared_ptr<ValidatedEmotionsClassificationNetwork> valid_network_;
  std::vector<Result> results_;
};

}

#endif //OPENVINO_PIPELINE_LIB_EMOTIONS_RECOGNITION_H
