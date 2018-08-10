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

#include "openvino_service/pipeline.h"

using namespace InferenceEngine;

Pipeline::Pipeline() {
  counter_ = 0;
}

bool Pipeline::add(const std::string &name,
                   std::shared_ptr<Input::BaseInputDevice> input_device) {
  input_device_name_ = name;
  input_device_ = std::move(input_device);
  next_.insert({"", name});
  return true;
};

bool Pipeline::add(const std::string &parent, const std::string &name,
                   std::shared_ptr<Outputs::BaseOutput> output) {
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
                   std::shared_ptr<openvino_service::BaseInference> inference) {
  if (name_to_detection_map_.find(parent) == name_to_detection_map_.end()
      && input_device_name_ != parent) {
    slog::err << "parent device/detection does not exists!" << slog::endl;
    return false;
  }
  next_.insert({parent, name});
  name_to_detection_map_[name] = std::move(inference);
  ++total_inference_;
  return true;
};

void Pipeline::runOnce() {
  counter_ = 0;
  if (!input_device_->read(&frame_)) {
    throw std::logic_error("Failed to get frame from cv::VideoCapture");
  }
  width_ = frame_.cols;
  height_ = frame_.rows;
  for (auto &pair: name_to_output_map_) {
    pair.second->feedFrame(frame_);
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  for (auto pos = next_.equal_range(input_device_name_);
       pos.first != pos.second; ++pos.first) {
    std::string detection_name = pos.first->second;
    auto detection_ptr = name_to_detection_map_[detection_name];
    detection_ptr->enqueue(
        frame_, cv::Rect(width_ / 2, height_ / 2, width_, height_));
    ++counter_;
    detection_ptr->submitRequest();
  }
  std::unique_lock<std::mutex> lock(counter_mutex_);
  cv_.wait(lock, [self = this]() { return self->counter_ == 0; });
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
  if (!input_device_->read(&frame_)) {
    throw std::logic_error("Failed to get frame from cv::VideoCapture");
  }
  width_ = frame_.cols;
  height_ = frame_.rows;
  for (auto &pair: name_to_output_map_) {
    pair.second->feedFrame(frame_);
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
          cv::Mat next_input = frame_(clippedRect);
          next_detection_ptr->enqueue(next_input, prev_result.location);
        }
        if (detection_ptr->getResultsLength() > 0) {
          ++counter_;
          next_detection_ptr->submitRequest();
        }
      }
    }
  }
  std::lock_guard<std::mutex> lk(counter_mutex_);
  --counter_;
  cv_.notify_all();
}
