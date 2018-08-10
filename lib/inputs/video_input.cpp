#include "openvino_service/inputs/video_input.h"

//Video
Input::Video::Video(const std::string &video) {
  video_.assign(video);
}

bool Input::Video::initialize() {
  setInitStatus(cap.open(video_));
  setWidth((size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH));
  setHeight((size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  return isInit();
}

bool Input::Video::initialize(size_t width, size_t height) {
  setWidth(width);
  setHeight(height);
  setInitStatus(cap.open(video_));
  if (isInit()) {
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
  }
  return isInit();
}

bool Input::Video::read(cv::Mat *frame) {
  if (!isInit()) { return false; }
  cap.grab();
  return cap.retrieve(*frame);
}

void Input::Video::config() {
  //TODO
}
