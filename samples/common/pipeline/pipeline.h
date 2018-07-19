//
// Created by chris on 18-7-19.
//

#ifndef SAMPLES_PIPELINE_H
#define SAMPLES_PIPELINE_H

#include <memory>

#include "input_devices/input.h"

class Pipeline {
public:
    Pipeline() = default;
    void add(const std::string &parent, const std::string &name, BaseInputDevice *input_device){};
    void doIferenceOnce(){};
    void doIferenceSpin(){};

private:
    std::shared_ptr<BaseInputDevice> input_device_;
    std::shared_ptr<BaseInputDevice> output_device_;
};


#endif //SAMPLES_PIPELINE_H
