//
// Created by chris on 18-7-11.
//

#ifndef SAMPLES_FACTORY_H
#define SAMPLES_FACTORY_H

#include "input.h"

class Factory {
public:
    static std::unique_ptr<BaseInputDevice> makeInputDevice(const std::string &);
};

#endif //SAMPLES_FACTORY_H
