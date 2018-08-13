/**
* @brief A header file with declaration for ImageWindowOutput Class
* @file image_window_output.h
*/

#ifndef OPENVINO_PIPELINE_LIB_IMAGE_WINDOW_OUTPUT_H
#define OPENVINO_PIPELINE_LIB_IMAGE_WINDOW_OUTPUT_H

#include "openvino_service/outputs/base_output.h"

namespace Outputs {

class ImageWindowOutput : public BaseOutput {
 public:
  explicit ImageWindowOutput(const std::string &window_name,
                             int focal_length = 950);
  void feedFrame(const cv::Mat &) override;
  void handleOutput(const std::string &overall_output_text) override;
  void accept(const openvino_service::Result&) override ;

 private:
  const std::string window_name_;
  cv::Mat frame_;
  float focal_length_;
  cv::Mat camera_matrix_;
};

}
#endif //OPENVINO_PIPELINE_LIB_IMAGE_WINDOW_OUTPUT_H
