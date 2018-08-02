//
// Created by chris on 18-7-19.
//

#ifndef SAMPLES_PIPELINE_H
#define SAMPLES_PIPELINE_H

#include "detection.h"

#include "io_devices/input.h"
#include "io_devices/output.h"
#include "samples/slog.hpp"

#include "opencv2/opencv.hpp"

#include <memory>
#include <atomic>
#include <mutex>
#include <future>

class Pipeline {
 public:
  Pipeline() = default;
  bool add(const std::string &parent, const std::string &name,
           std::shared_ptr<BaseInputDevice> input_device);
  bool add(const std::string &parent, const std::string &name,
           std::shared_ptr<DetectionClass::Detection> detection);
  bool add(const std::string &parent, const std::string &name,
           std::shared_ptr<BaseOutput> output);
  bool add(const std::string &parent, const std::string &name);
  void runOnce();
  void callback(const std::string &detection_name);
  void setcallback();

 private:
  std::shared_ptr<BaseInputDevice> input_device_;
  std::string input_device_name_;
  std::multimap<std::string, std::string> next_;
  std::map<std::string, std::shared_ptr<DetectionClass::Detection>>
      name_to_detection_map_;
  std::map<std::string, std::shared_ptr<BaseOutput>> name_to_output_map_;
  void printPipeline();
  int total_detection_ = 0;
  std::set<std::string> output_names_;
  int width_;
  int height_;
  cv::Mat frame;
  // for multi threads
  std::atomic<int> counter;
  std::mutex counter_mutex;
  std::condition_variable cv;
};

#endif //SAMPLES_PIPELINE_H
