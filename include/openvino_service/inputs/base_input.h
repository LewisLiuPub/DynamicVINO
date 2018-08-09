//
// Created by chris on 18-8-9.
//

#ifndef OPENVINO_PIPELINE_LIB_BASE_INPUT_H
#define OPENVINO_PIPELINE_LIB_BASE_INPUT_H

#include <opencv2/opencv.hpp>

/**
* @class BaseInputDevice
* @brief This class is an interface for three kinds of input devices: realsense camera, standard camera and video
*/
namespace Input {
class BaseInputDevice {
 public:
  /**
  * @brief initialize the input device, for cameras, it will turn the camera on and get ready to read frames,
   * for videos, it will open a video file
  * @return whether the input device is successfully turned on
  */
  virtual bool initialize() = 0;

  virtual bool initialize(int) = 0;

  virtual bool initialize(size_t width, size_t height) = 0;
  /**
  * @brief read next frame, and give the value to argument frame
  * @return whether the next frame is successfully read
  */
  virtual bool read(cv::Mat *frame) = 0;
  virtual void config() = 0; //< TODO
  virtual ~BaseInputDevice() = default;
  inline size_t getWidth() { return width_; }
  inline void setWidth(size_t width) { width_ = width; }
  inline size_t getHeight() { return height_; }
  inline void setHeight(size_t height) { height_ = height; }
  inline bool getIsInit() { return is_init_; }
  inline void setIsInit(bool is_init) { is_init_ = is_init; }
 private:
  size_t width_ = 0;
  size_t height_ = 0;
  bool is_init_ = false;
};
}
#endif //OPENVINO_PIPELINE_LIB_BASE_INPUT_H
