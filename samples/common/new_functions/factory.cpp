//
// Created by chris on 18-7-11.
//
#include <memory>

#include "factory.h"

std::unique_ptr<BaseInputDevice> Factory::makeInputDevice(const std::string &FLAGS_i) {
    if (FLAGS_i == "RealSenseCamera") {
        return std::unique_ptr<RealSenseCamera>(new RealSenseCamera());
    }
    else if (FLAGS_i == "StandardCamera") {
        return std::unique_ptr<StandardCamera>(new StandardCamera());
    } else {
        return std::unique_ptr<Video>(new Video(FLAGS_i));
    }
}