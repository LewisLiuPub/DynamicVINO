/**
 * @brief A header file with declaration for HeadPoseDetectionModel Class
 * @file head_pose_detection_model.h
 */
#ifndef OPENVINO_PIPELINE_LIB_BASE_OUTPUT_H
#define OPENVINO_PIPELINE_LIB_BASE_OUTPUT_H

#include "opencv2/opencv.hpp"
#include "openvino_service/inferences/base_inference.h"

namespace Outputs {
/**
 * @class BaseOutput
 * @brief This class is a base class for various output devices. It employs
 * visitor pattern to perform different operations to different inference
 * result with different output device
 */
class BaseOutput {
 public:
  BaseOutput() = default;
  virtual void accept(const openvino_service::Result&) = 0;
  virtual void feedFrame(const cv::Mat &frame) {}
  virtual void handleOutput(const std::string &overall_output_text) = 0;
};

}
#endif //OPENVINO_PIPELINE_LIB_BASE_OUTPUT_H
