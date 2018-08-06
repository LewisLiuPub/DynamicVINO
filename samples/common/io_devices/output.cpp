//
// Created by chris on 18-7-20.
//

#include "output.h"

void ImageWindow::feedFrame(const cv::Mat &frame) {
  frame_ = frame;
}

void ImageWindow::prepareData(
    const InferenceResult::FaceDetectionResult &result) {
  std::ostringstream out;
  cv::Rect rect = result.location;

  out.str("");
  if (result.confidence >= 0) {
    out << "face detection confidence: "
        << result.label << ": "
        << std::fixed << std::setprecision(3)
        << result.confidence << ",";
  }
  cv::putText(frame_,
              out.str(),
              cv::Point2f(result.location.x, result.location.y - 15),
              cv::FONT_HERSHEY_COMPLEX_SMALL,
              0.8,
              cv::Scalar(0, 0, 255));
  cv::rectangle(frame_, result.location, cv::Scalar(100, 100, 100), 1);
}

void ImageWindow::prepareData(
    const InferenceResult::EmotionsDetectionResult &result) {
  std::ostringstream out;
  cv::Rect rect = result.location;

  out.str("");
  out << "emotions detection confidence: " <<
      result.label << ": " <<
      std::fixed << std::setprecision(3) <<
      result.confidence << ",";
  cv::putText(frame_,
              out.str(),
              cv::Point2f(result.location.x, result.location.y + 15),
              cv::FONT_HERSHEY_COMPLEX_SMALL,
              0.8,
              cv::Scalar(0, 0, 255));
  cv::rectangle(frame_, result.location, cv::Scalar(100, 100, 100), 1);
}

void ImageWindow::prepareData(
    const InferenceResult::AgeGenderResult &result) {
  std::ostringstream out;
  cv::Rect rect = result.location;

  out.str("");
  out << "age gender detection confidence: " <<
      "age is: " << result.age << "," <<
      "gender is: " << (result.male_prob > 0.5)? "M":"F";
  cv::putText(frame_,
              out.str(),
              //cv::Point2f(result.location.x, result.location.y),
              cv::Point2f(0,30),
              cv::FONT_HERSHEY_COMPLEX_SMALL,
              0.8,
              cv::Scalar(0, 0, 255));
  cv::rectangle(frame_, result.location, cv::Scalar(100, 100, 100), 1);
}

void ImageWindow::handleOutput(const std::string &overall_output_text) {
  cv::putText(frame_,
              overall_output_text,
              cv::Point2f(0, 65),
              cv::FONT_HERSHEY_TRIPLEX,
              0.5,
              cv::Scalar(255, 0, 0));
  cv::imshow(window_name_, frame_);
}