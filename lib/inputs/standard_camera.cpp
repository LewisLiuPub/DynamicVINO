//
// Created by chris on 18-8-9.
//

#include "openvino_service/inputs/standard_camera.h"

//StandardCamera
bool Input::StandardCamera::initialize() {
  setIsInit(cap.open(0));
  setWidth((size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH));
  setHeight((size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  return getIsInit();
}
bool Input::StandardCamera::initialize(int camera_num) {
  setIsInit(cap.open(camera_num));
  setWidth((size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH));
  setHeight((size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  return getIsInit();
}
bool Input::StandardCamera::initialize(size_t width, size_t height) {
  setWidth(width);
  setHeight(height);
  setIsInit(cap.open(0));
  if (getIsInit()) {
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
  }
  return getIsInit();
}
bool Input::StandardCamera::read(cv::Mat *frame) {
  if (!getIsInit()) { return false; }
  cap.grab();
  return cap.retrieve(*frame);
}
void Input::StandardCamera::config() {
  //TODO
}