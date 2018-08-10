#include "openvino_service/outputs/image_window_output.h"

Outputs::ImageWindowOutput::ImageWindowOutput(
    const std::string &window_name, int focal_length) :
    window_name_(window_name), focal_length_(focal_length) {
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
}

void Outputs::ImageWindowOutput::feedFrame(const cv::Mat &frame) {
  //frame_ = frame;
  frame_ = frame.clone();
  if (camera_matrix_.empty()) {
    int cx = frame.cols / 2;
    int cy = frame.rows / 2;
    camera_matrix_ = cv::Mat::zeros(3, 3, CV_32F);
    camera_matrix_.at<float>(0) = focal_length_;
    camera_matrix_.at<float>(2) = static_cast<float>(cx);
    camera_matrix_.at<float>(4) = focal_length_;
    camera_matrix_.at<float>(5) = static_cast<float>(cy);
    camera_matrix_.at<float>(8) = 1;
  }
}

void Outputs::ImageWindowOutput::accept(const openvino_service::Result& result) {
  result.decorateFrame(&frame_, &camera_matrix_);
};

void Outputs::ImageWindowOutput::handleOutput(
    const std::string &overall_output_text) {
  cv::putText(frame_,
              overall_output_text,
              cv::Point2f(0, 65),
              cv::FONT_HERSHEY_TRIPLEX,
              0.5,
              cv::Scalar(255, 0, 0));
  cv::imshow(window_name_, frame_);
}