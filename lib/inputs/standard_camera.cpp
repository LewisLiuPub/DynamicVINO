#include "openvino_service/inputs/standard_camera.h"

//StandardCamera
bool Input::StandardCamera::initialize() {
  setInitStatus(cap.open(0));
  setWidth((size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH));
  setHeight((size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  return isInit();
}

bool Input::StandardCamera::initialize(int camera_num) {
  setInitStatus(cap.open(camera_num));
  setWidth((size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH));
  setHeight((size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  return isInit();
}

bool Input::StandardCamera::initialize(size_t width, size_t height) {
  setWidth(width);
  setHeight(height);
  setInitStatus(cap.open(0));
  if (isInit()) {
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
  }
  return isInit();
}

bool Input::StandardCamera::read(cv::Mat *frame) {
  if (!isInit()) { return false; }
  cap.grab();
  return cap.retrieve(*frame);
}

void Input::StandardCamera::config() {
  //TODO
}