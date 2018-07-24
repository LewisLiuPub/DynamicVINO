//
// Created by chris on 18-7-19.
//

#ifndef SAMPLES_PIPELINE_H
#define SAMPLES_PIPELINE_H

#include "io_devices/input.h"
#include "io_devices/output.h"
#include "samples/slog.hpp"
#include "detection.h"

#include "opencv2/opencv.hpp"

#include <memory>


class Pipeline {
public:
    Pipeline() = default;
    bool add(const std::string &parent, const std::string &name,
                       std::unique_ptr<BaseInputDevice> input_device);
    bool add(const std::string &parent, const std::string &name,
                       std::unique_ptr<DetectionClass::Detection> detection);
    bool add(const std::string &parent, const std::string &name,
                       std::unique_ptr<BaseOutput> output);
    void runOnce();

private:
    void requestCallback(InferenceEngine::IInferRequest::Ptr context, InferenceEngine::StatusCode code);
    std::unique_ptr<BaseInputDevice> input_device_;
    std::string input_device_name_;
    std::multimap<std::string, std::string> next_;
    std::map<std::string, std::unique_ptr<DetectionClass::Detection>> name_to_detection_map_;
    void printPipeline();
    std::shared_ptr<BaseOutput> output_;
    std::string output_name_;
};


#endif //SAMPLES_PIPELINE_H
