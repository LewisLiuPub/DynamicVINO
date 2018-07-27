//
// Created by chris on 18-7-19.
//
#include "pipeline.h"

using namespace InferenceEngine;

bool Pipeline::add(const std::string &parent, const std::string &name,
                   std::unique_ptr<BaseInputDevice> input_device) {
    if (!parent.empty()) {
        slog::err<<"input device should have no parent!"<<slog::endl;
        return false;
    }
    input_device_name_ = name;
    input_device_ = std::move(input_device);
    next_.insert({parent, name});
    return true;
};

bool Pipeline::add(const std::string &parent, const std::string &name,
                   std::unique_ptr<BaseOutput> output) {
    if (parent.empty()) {
        slog::err<<"output device have no parent!"<<slog::endl;
        return false;
    }
    if (name_to_detection_map_.find(parent) == name_to_detection_map_.end()) {
        slog::err<<"parent detection does not exists!"<<slog::endl;
        return false;
    }
    output_names_.emplace_back(name);
    outputs_.emplace_back(std::move(output));
    next_.insert({parent, name});
    return true;
};

bool Pipeline::add(const std::string &parent, const std::string &name) {
    if (parent.empty()) {
        slog::err<<"output device should have no parent!"<<slog::endl;
        return false;
    }
    if (name_to_detection_map_.find(parent) == name_to_detection_map_.end()) {
        slog::err<<"parent detection does not exists!"<<slog::endl;
        return false;
    }
    if (std::find(output_names_.begin(), output_names_.end(), name) == output_names_.end())  {
        slog::err<<"output does not exists!"<<slog::endl;
        return false;
    }
    next_.insert({parent, name});
    return true;
}

bool Pipeline::add(const std::string &parent, const std::string &name,
                   std::unique_ptr<DetectionClass::Detection> detection) {
    if (name_to_detection_map_.find(parent) == name_to_detection_map_.end() && input_device_name_ != parent) {
        slog::err<<"parent device/detection does not exists!"<<slog::endl;
        return false;
    }
    next_.insert({parent, name});
    name_to_detection_map_[name] = std::move(detection);
    ++total_detection_;
    return true;
};

void Pipeline::runOnce() {
    cv::Mat frame;
    std::list<std::string> detection_list;
    //slog::info << "Reading input" << slog::endl;
    std::atomic<int> counter(0);
    std::mutex counter_mutex;
    std::condition_variable cv;
    const auto &next = next_;
    auto &outputs_alias = outputs_;
    auto& name_to_detection_map_alias = name_to_detection_map_;

    if (!input_device_->read(&frame)) {
        throw std::logic_error("Failed to get frame from cv::VideoCapture");
    }
    for (auto& output_ptr: outputs_) {
        if (auto face_detection_output = dynamic_cast<ImageWindow*>(output_ptr.get())) {
            face_detection_output->feedFrame(frame);
        }
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto pos = next_.equal_range(input_device_name_);
            pos.first != pos.second; ++pos.first) {
        std::string detection_name = pos.first->second;
        auto detection_ptr = name_to_detection_map_[detection_name];
        detection_ptr->enqueue(frame);
        int width = frame.cols;
        int height = frame.rows;
        std::function<void(void)> callback;
        callback = [detection_ptr, detection_name, &next, &outputs_alias, &frame,
                    &callback, width, height, &counter, &cv,
                &name_to_detection_map_alias, &counter_mutex]() mutable {
            //slog::info<<"Hello callback"<<slog::endl;
            detection_ptr->fetchResults();
            // set output
            for (size_t i = 0; i < detection_ptr->getResultsLength(); ++i) {
                for (auto &output_ptr : outputs_alias) {
                    output_ptr->prepareData((*detection_ptr)[i]);
                }
            }
            // set input for next network
            auto detection_ptr_alias = detection_ptr;
            auto detection_name_alias = detection_name;
            for (auto next_pos = next.equal_range(detection_name_alias);
                 next_pos.first != next_pos.second; ++next_pos.first) {
                detection_name = next_pos.first->second;
                auto detection_ptr_iter = name_to_detection_map_alias.find(detection_name);
                if (detection_ptr_iter != name_to_detection_map_alias.end()) {
                    detection_ptr = detection_ptr_iter->second;
                    for (size_t i = 0; i < detection_ptr_alias->getResultsLength(); ++i) {
                        DetectionClass::Detection::Result prev_result = (*detection_ptr_alias)[i];
                        auto clippedRect = prev_result.location & cv::Rect(0, 0,
                                                                           width,
                                                                           height);
                        cv::Mat next_input = frame(clippedRect);
                        detection_ptr->enqueue(next_input);
                        //TODO: add set bounding box
                        //detection_ptr->setBoundingBox(prev_result.boundingbox)
                    }
                    counter++;
                    detection_ptr->setCompletionCallback(callback);
                    detection_ptr->submitRequest();
                }
            }
            std::lock_guard<std::mutex> lk(counter_mutex);
            counter--;
            //slog::info<<counter<<slog::endl;
            cv.notify_all();
        };
        detection_ptr->setCompletionCallback(callback);
        counter++;
        detection_ptr->submitRequest();
    }
    std::unique_lock<std::mutex> lock(counter_mutex);
    cv.wait(lock, [&counter](){ return counter == 0; });
    auto t1 = std::chrono::high_resolution_clock::now();
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    //calculate fps
    ms secondDetection = std::chrono::duration_cast<ms>(t1 - t0);
    std::ostringstream out;
    std::string window_output_string = "(" + std::to_string(1000.f / secondDetection.count()) + " fps)";
    for (auto &output : outputs_) {
        output->handleOutput(window_output_string);
    }
}

void Pipeline::printPipeline() {
    for (auto &current_node : next_) {
        printf("Name: %s --> Name: %s", current_node.first.c_str(), current_node.second.c_str());
    }
}
