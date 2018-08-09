//
// Created by chris on 18-8-9.
//

#ifndef OPENVINO_PIPELINE_LIB_FACE_DETECTION_H
#define OPENVINO_PIPELINE_LIB_FACE_DETECTION_H

#include <memory>

#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"

#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/engines/engine.h"
#include "openvino_service/data_struct.h"
#include "openvino_service/outputs/output.h"

namespace openvino_service {

class FaceDetection : public BaseInference {
 public:
  using Result = InferenceResult::FaceDetectionResult;
  explicit FaceDetection(int, int, double);
  ~FaceDetection() override;
  void loadNetwork(std::shared_ptr<ValidatedFaceDetectionNetwork>);
  bool enqueue(const cv::Mat &, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<BaseOutput> output_visitor) override;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override;
  const std::string getName() const override;

 private:
  std::shared_ptr<ValidatedFaceDetectionNetwork> valid_network_;
  std::vector<Result> results_;
  int width_ = 0;
  int height_ = 0;
  int max_proposal_count_;
  int object_size_;
  double show_output_thresh_ = 0;
};

}
#endif //OPENVINO_PIPELINE_LIB_FACE_DETECTION_H
