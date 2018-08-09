//
// Created by chris on 18-7-20.
//
/**
 * @brief a header file with definition of Output class
 * @file output.h
 */
#ifndef SAMPLES_OUTPUT_H
#define SAMPLES_OUTPUT_H

#include "opencv2/opencv.hpp"
#include "result.h"
/**
 * @class BaseOutput
 * @brief This class is a base class for various output devices. It employs
 * visitor pattern to perform different operations to different inference
 * result with different output device
 */
class BaseOutput {
 public:
  BaseOutput() = default;
  virtual void prepareData(const InferenceResult::FaceDetectionResult&) = 0;
  virtual void prepareData(const InferenceResult::EmotionsResult&) = 0;
  virtual void prepareData(const InferenceResult::AgeGenderResult&) = 0;
  virtual void prepareData(const InferenceResult::HeadPoseResult&) = 0;
  virtual void handleOutput(const std::string &overall_output_text) = 0;
  virtual void feedFrame(const cv::Mat &frame) {}
};

class ImageWindow : public BaseOutput {
 public:
  explicit ImageWindow(const std::string &window_name, int);
  void prepareData(const InferenceResult::FaceDetectionResult&) override;
  void prepareData(const InferenceResult::EmotionsResult&) override;
  void prepareData(const InferenceResult::AgeGenderResult&) override;
  void prepareData(const InferenceResult::HeadPoseResult&) override;
  void feedFrame(const cv::Mat &) override;
  void handleOutput(const std::string &overall_output_text) override;

 private:
  const std::string window_name_;
  cv::Mat frame_;
  float focal_length_;
  cv::Mat camera_matrix_;
};

#endif //SAMPLES_OUTPUT_H
