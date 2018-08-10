/*
 * Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "openvino_service/outputs/image_window_output.h"

Outputs::ImageWindowOutput::ImageWindowOutput(
    const std::string &window_name, int focal_length) :
    window_name_(window_name), focal_length_(focal_length) {
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
}

void Outputs::ImageWindowOutput::feedFrame(const cv::Mat &frame) {
  frame_ = frame;
}

void Outputs::ImageWindowOutput::prepareData(
    const InferenceResult::FaceDetectionResult &result) {
  std::ostringstream out;
  cv::Rect rect = result.location;

  out.str("");
  if (result.confidence >= 0) {
    out << "Face Detection Confidence: "
        << std::fixed << std::setprecision(3)
        << result.confidence;
  }
  cv::putText(frame_,
              out.str(),
              cv::Point2f(result.location.x, result.location.y - 15),
              cv::FONT_HERSHEY_COMPLEX_SMALL,
              0.8,
              cv::Scalar(0, 0, 255));
  cv::rectangle(frame_, result.location, cv::Scalar(100, 100, 100), 1);
}

void Outputs::ImageWindowOutput::prepareData(
    const InferenceResult::EmotionsResult &result) {
  std::ostringstream out;
  cv::Rect rect = result.location;

  out.str("");
  out << "Emotions: " <<
      result.label << ": ";
  cv::putText(frame_,
              out.str(),
              cv::Point2f(result.location.x, result.location.y - 30),
              cv::FONT_HERSHEY_COMPLEX_SMALL,
              0.8,
              cv::Scalar(0, 0, 255));
  cv::rectangle(frame_, result.location, cv::Scalar(100, 100, 100), 1);
}

void Outputs::ImageWindowOutput::prepareData(
    const InferenceResult::AgeGenderResult &result) {
  std::ostringstream out;
  cv::Rect rect = result.location;

  out.str("");
  out << "Age is: " << result.age << "," <<
      "Gender is: " << ((result.male_prob > 0.5)?"M":"F");
  cv::putText(frame_,
              out.str(),
              cv::Point2f(result.location.x, result.location.y + 15),
              cv::FONT_HERSHEY_COMPLEX_SMALL,
              0.8,
              cv::Scalar(0, 0, 255));
  cv::rectangle(frame_, result.location, cv::Scalar(100, 100, 100), 1);
}

void Outputs::ImageWindowOutput::prepareData(
    const InferenceResult::HeadPoseResult& result) {
  int scale  = 50;
  int cx = frame_.cols / 2;
  int cy = frame_.rows / 2;
  double yaw = result.angle_y * CV_PI / 180.0;
  double pitch = result.angle_p * CV_PI / 180.0;
  double roll = result.angle_r * CV_PI / 180.0;
  cv::Rect rect = result.location;
  cv::Point3f cpoint(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
  cv::Matx33f Rx(1, 0, 0,
                 0, cos(pitch), -sin(pitch),
                 0, sin(pitch), cos(pitch));
  cv::Matx33f Ry(cos(yaw), 0, -sin(yaw),
                 0, 1, 0,
                 sin(yaw), 0, cos(yaw));
  cv::Matx33f Rz(cos(roll), -sin(roll), 0,
                 sin(roll), cos(roll), 0,
                 0, 0, 1);

  auto r = cv::Mat(Rz * Ry * Rx);

  if (camera_matrix_.empty()) {
    camera_matrix_ = cv::Mat::zeros(3, 3, CV_32F);
    camera_matrix_.at<float>(0) = focal_length_;
    camera_matrix_.at<float>(2) = static_cast<float>(cx);
    camera_matrix_.at<float>(4) = focal_length_;
    camera_matrix_.at<float>(5) = static_cast<float>(cy);
    camera_matrix_.at<float>(8) = 1;
  }

  cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F),
      zAxis1(3, 1, CV_32F);

  xAxis.at<float>(0) = 1 * scale;
  xAxis.at<float>(1) = 0;
  xAxis.at<float>(2) = 0;

  yAxis.at<float>(0) = 0;
  yAxis.at<float>(1) = -1 * scale;
  yAxis.at<float>(2) = 0;

  zAxis.at<float>(0) = 0;
  zAxis.at<float>(1) = 0;
  zAxis.at<float>(2) = -1 * scale;

  zAxis1.at<float>(0) = 0;
  zAxis1.at<float>(1) = 0;
  zAxis1.at<float>(2) = 1 * scale;

  cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
  o.at<float>(2) = camera_matrix_.at<float>(0);

  xAxis = r * xAxis + o;
  yAxis = r * yAxis + o;
  zAxis = r * zAxis + o;
  zAxis1 = r * zAxis1 + o;

  cv::Point p1, p2;

  p2.x = static_cast<int>(
      (xAxis.at<float>(0) / xAxis.at<float>(2) * camera_matrix_.at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (xAxis.at<float>(1) / xAxis.at<float>(2) * camera_matrix_.at<float>(4))
          + cpoint.y);
  cv::line(frame_, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

  p2.x = static_cast<int>(
      (yAxis.at<float>(0) / yAxis.at<float>(2) * camera_matrix_.at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (yAxis.at<float>(1) / yAxis.at<float>(2) * camera_matrix_.at<float>(4))
          + cpoint.y);
  cv::line(frame_, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

  p1.x = static_cast<int>(
      (zAxis1.at<float>(0) / zAxis1.at<float>(2) * camera_matrix_.at<float>(0))
          + cpoint.x);
  p1.y = static_cast<int>(
      (zAxis1.at<float>(1) / zAxis1.at<float>(2) * camera_matrix_.at<float>(4))
          + cpoint.y);

  p2.x = static_cast<int>(
      (zAxis.at<float>(0) / zAxis.at<float>(2) * camera_matrix_.at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (zAxis.at<float>(1) / zAxis.at<float>(2) * camera_matrix_.at<float>(4))
          + cpoint.y);
  cv::line(frame_, p1, p2, cv::Scalar(255, 0, 0), 2);
  cv::circle(frame_, p2, 3, cv::Scalar(255, 0, 0), 2);
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