//
// Created by chris on 18-8-9.
//

#ifndef OPENVINO_PIPELINE_LIB_HEAD_POSE_RECOGNITION_H
#define OPENVINO_PIPELINE_LIB_HEAD_POSE_RECOGNITION_H

#include <memory>

#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"

#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/engines/engine.h"
#include "openvino_service/data_struct.h"
#include "openvino_service/outputs/output.h"

namespace openvino_service {

// Head Pose Detection
class HeadPoseDetection : public BaseInference {
 public:
  using Result = InferenceResult::HeadPoseResult;
  explicit HeadPoseDetection();
  ~HeadPoseDetection() override;
  void loadNetwork(std::shared_ptr<ValidatedHeadPoseNetwork>);
  bool enqueue(const cv::Mat &frame, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<BaseOutput> output_visitor) override;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override;
  const std::string getName() const override;

 private:
  std::shared_ptr<ValidatedHeadPoseNetwork> valid_network_;
  std::vector<Result> results_;
};

}
#endif //OPENVINO_PIPELINE_LIB_HEAD_POSE_RECOGNITION_H
