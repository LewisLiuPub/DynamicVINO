#include "openvino_service/inputs/realsense_camera.h"

#include "openvino_service/slog.hpp"

//RealSenseCamera
bool Input::RealSenseCamera::initialize() {
  cfg_.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
  setInitStatus(pipe_.start(cfg_));
  setWidth(640);
  setHeight(480);
  if (!isInit()) { return false; }
  if (first_read_) {
    rs2::frameset frames;
    for (int i = 0; i < 30; i++) {
      //Wait for all configured streams to produce a frame
      try {
        frames = pipe_.wait_for_frames();
      } catch (...) {
        return false;
      }
    }
    first_read_ = false;
  }
  return true;
}
bool Input::RealSenseCamera::initialize(size_t width, size_t height) {
  if (3 * width != 4 * height) {
    slog::err << "The aspect ratio must be 4:3 when using RealSense camera"
              << slog::endl;
    return false;
  }
  cfg_.enable_stream(RS2_STREAM_COLOR,
                     (int) width,
                     (int) height,
                     RS2_FORMAT_BGR8,
                     30);
  setInitStatus(pipe_.start(cfg_));
  setWidth(width);
  setHeight(height);
  if (!isInit()) { return false; }
  if (first_read_) {
    rs2::frameset frames;
    for (int i = 0; i < 30; i++) {
      //Wait for all configured streams to produce a frame
      try {
        frames = pipe_.wait_for_frames();
      } catch (...) {
        return false;
      }
    }
    first_read_ = false;
  }
  return true;
}
bool Input::RealSenseCamera::read(cv::Mat *frame) {
  if (!isInit()) { return false; }
  rs2::frameset data =
      pipe_.wait_for_frames(); // Wait for next set of frames from the camera
  rs2::frame color_frame;
  try {
    color_frame = data.get_color_frame();
  } catch (...) {
    return false;
  }
  cv::Mat(cv::Size((int) getWidth(), (int) getHeight()),
          CV_8UC3,
          (void *) color_frame.get_data(),
          cv::Mat::AUTO_STEP).copyTo(*frame);
  return true;
}
void Input::RealSenseCamera::config() {
  //TODO
}