//
// Created by chris on 18-7-19.
//
#include "pipeline.h"

using namespace InferenceEngine;

bool Pipeline::add(const std::string &parent, const std::string &name,
                   std::shared_ptr<BaseInputDevice> input_device) {
  if (!parent.empty()) {
    slog::err << "input device should have no parent!" << slog::endl;
    return false;
  }
  input_device_name_ = name;
  input_device_ = std::move(input_device);
  next_.insert({parent, name});
  return true;
};

bool Pipeline::add(const std::string &parent, const std::string &name,
                   std::shared_ptr<BaseOutput> output) {
  if (parent.empty()) {
    slog::err << "output device have no parent!" << slog::endl;
    return false;
  }
  if (name_to_detection_map_.find(parent) == name_to_detection_map_.end()) {
    slog::err << "parent detection does not exists!" << slog::endl;
    return false;
  }
  output_names_.insert(name);
  name_to_output_map_[name] = std::move(output);
  next_.insert({parent, name});
  return true;
};

bool Pipeline::add(const std::string &parent, const std::string &name) {
  if (parent.empty()) {
    slog::err << "output device should have no parent!" << slog::endl;
    return false;
  }
  if (name_to_detection_map_.find(parent) == name_to_detection_map_.end()) {
    slog::err << "parent detection does not exists!" << slog::endl;
    return false;
  }
  if (std::find(output_names_.begin(), output_names_.end(), name)
      == output_names_.end()) {
    slog::err << "output does not exists!" << slog::endl;
    return false;
  }
  next_.insert({parent, name});
  return true;
}

bool Pipeline::add(const std::string &parent, const std::string &name,
                   std::shared_ptr<openvino_service::BaseInference> detection) {
  if (name_to_detection_map_.find(parent) == name_to_detection_map_.end()
      && input_device_name_ != parent) {
    slog::err << "parent device/detection does not exists!" << slog::endl;
    return false;
  }
  next_.insert({parent, name});
  name_to_detection_map_[name] = std::move(detection);
  ++total_detection_;
  return true;
};

void Pipeline::runOnce() {
  counter = 0;
  if (!input_device_->read(&frame)) {
    throw std::logic_error("Failed to get frame from cv::VideoCapture");
  }
  width_ = frame.cols;
  height_ = frame.rows;
  for (auto &pair: name_to_output_map_) {
    pair.second->feedFrame(frame);
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  for (auto pos = next_.equal_range(input_device_name_);
       pos.first != pos.second; ++pos.first) {
    std::string detection_name = pos.first->second;
    auto detection_ptr = name_to_detection_map_[detection_name];
    detection_ptr->enqueue(
        frame, cv::Rect(width_ / 2, height_ / 2, width_, height_));
    ++counter;
    detection_ptr->submitRequest();
  }
  std::unique_lock<std::mutex> lock(counter_mutex);
  cv.wait(lock, [self = this]() { return self->counter == 0; });
  auto t1 = std::chrono::high_resolution_clock::now();
  typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
  //calculate fps
  ms secondDetection = std::chrono::duration_cast<ms>(t1 - t0);
  std::ostringstream out;
  std::string window_output_string =
      "(" + std::to_string(1000.f / secondDetection.count()) + " fps)";
  for (auto &pair : name_to_output_map_) {
    pair.second->handleOutput(window_output_string);
  }
}

void Pipeline::printPipeline() {
  for (auto &current_node : next_) {
    printf("Name: %s --> Name: %s",
           current_node.first.c_str(),
           current_node.second.c_str());
  }
}

void Pipeline::setcallback() {
  if (!input_device_->read(&frame)) {
    throw std::logic_error("Failed to get frame from cv::VideoCapture");
  }
  width_ = frame.cols;
  height_ = frame.rows;
  for (auto &pair: name_to_output_map_) {
    pair.second->feedFrame(frame);
  }
  for (auto &pair: name_to_detection_map_) {
    std::string detection_name = pair.first;
    std::function<void(void)> callb;
    callb = [detection_name, self = this]() {
      self->callback(detection_name);
      return;
    };
    pair.second->getEngine()->getRequest()->SetCompletionCallback(callb);
  }
}
void Pipeline::callback(const std::string &detection_name) {
  //slog::info<<"Hello callback"<<slog::endl;
  auto detection_ptr = name_to_detection_map_[detection_name];
  detection_ptr->fetchResults();
  // set output
  for (auto pos = next_.equal_range(detection_name);
       pos.first != pos.second; ++pos.first) {
    std::string next_name = pos.first->second;
    // if next is output, then print
    if (output_names_.find(next_name) != output_names_.end()) {
      detection_ptr->accepts(name_to_output_map_[next_name]);
    }
    // if next is network, set input for next network
    else {
      auto detection_ptr_iter = name_to_detection_map_.find(
          next_name);
      if (detection_ptr_iter != name_to_detection_map_.end()) {
        auto next_detection_ptr = detection_ptr_iter->second;
        for (size_t i = 0; i < detection_ptr->getResultsLength(); ++i) {
          InferenceResult::Result prev_result =
              detection_ptr->getLocationResult(i);
          auto clippedRect = prev_result.location & cv::Rect(0, 0,
                                                             width_,
                                                             height_);
          cv::Mat next_input = frame(clippedRect);
          next_detection_ptr->enqueue(next_input, prev_result.location);
        }
        if (detection_ptr->getResultsLength() > 0) {
          ++counter;
          next_detection_ptr->submitRequest();
        }
      }
    }
  }
  std::lock_guard<std::mutex> lk(counter_mutex);
  --counter;
  cv.notify_all();
}
