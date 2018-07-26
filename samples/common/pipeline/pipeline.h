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
                       std::unique_ptr<BaseInputDevice> input_device);
    bool add(const std::string &parent, const std::string &name,
                       std::unique_ptr<DetectionClass::Detection> detection);
    bool add(const std::string &parent, const std::string &name,
                       std::unique_ptr<BaseOutput> output);
    bool add(const std::string &parent, const std::string &name);
    void runOnce();

private:
    std::unique_ptr<BaseInputDevice> input_device_;
    std::string input_device_name_;
    std::multimap<std::string, std::string> next_;
    std::map<std::string, std::shared_ptr<DetectionClass::Detection>> name_to_detection_map_;
    void printPipeline();
    std::vector<std::shared_ptr<BaseOutput>> outputs_;
    int total_detection_ = 0;
    std::vector<std::string> output_names_;
};


#endif //SAMPLES_PIPELINE_H
