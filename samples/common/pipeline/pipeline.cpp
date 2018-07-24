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
        slog::err<<"output device should have no parent!"<<slog::endl;
        return false;
    }
    if (name_to_detection_map_.find(parent) == name_to_detection_map_.end()) {
        slog::err<<"parent does not exists!"<<slog::endl;
        return false;
    }
    output_name_ = name;
    output_= std::move(output);
    next_.insert({parent, name});
    return true;
};

bool Pipeline::add(const std::string &parent, const std::string &name,
                   std::unique_ptr<DetectionClass::Detection> detection) {
    if (name_to_detection_map_.find(parent) == name_to_detection_map_.end()) {
        slog::err<<"parent does not exists!"<<slog::endl;
        return false;
    }
    next_.insert({parent, name});
    name_to_detection_map_[name] = std::move(detection);
    return true;
};


void Pipeline::runOnce() {
    cv::Mat frame;
    std::list<std::string> detection_list;
    slog::info << "Reading input" << slog::endl;
    if (!input_device_->read(&frame)) {
        throw std::logic_error("Failed to get frame from cv::VideoCapture");
    }
    if (auto face_detection_output = dynamic_cast<ImageWindow*>(output_.get())) {
        face_detection_output->feedFrame(frame);
    }

    for (auto pos = next_.equal_range(input_device_name_);
            pos.first != pos.second; ++pos.first) {
        std::string detection_name = pos.first->second;
        auto &detection_ptr = name_to_detection_map_[detection_name];
        detection_ptr->enqueue(frame);
        // define lambda object
        auto callback = [&detection_ptr, &detection_name, &next_]{
            slog::info<<"Hello callback"<<slog::endl;
            detection_ptr->fetchResults();
            for (size_t i = 0; i < detection_ptr->getResultsLength(); ++i) {
                output_->prepareData(*(detection_ptr->getResultPtr(i)));
            }
            for (auto next_pos = next_.equal_range(detection_name);
                 next_pos.first != next_pos.second; ++next_pos.first) {
                std::string next_detection_name = pos.first->second;
                detection_ptr->setCompletionCallback(callback);
                detection_ptr->submitRequest();
            }
            //int *a;
            //printf("the label of the first result is: %d", dynamic_cast<DetectionClass::FaceDetection*>(detection_ptr.get())->getResults(0).label);
        };
        detection_ptr->setCompletionCallback(callback);
        detection_ptr->submitRequest();

    }
}

void Pipeline::printPipeline() {
    for (auto &current_node : next_) {
        printf("Name: %s --> Name: %s", current_node.first.c_str(), current_node.second.c_str());
    }
}
