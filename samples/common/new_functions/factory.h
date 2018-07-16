//
// Created by chris on 18-7-11.
//

#ifndef SAMPLES_FACTORY_H
#define SAMPLES_FACTORY_H

#include "input.h"

/**
* @class Factory
* @brief This class is a factory class that produces the derived input device class corresponding to the input string
*/
class Factory {
public:
/**
  * @brief This function produces the derived input device class corresponding to the input string
  * @return the object of derived input device referenced by a smart pointer
  */
    static std::unique_ptr<BaseInputDevice> makeInputDevice(const std::string &);
};

#endif //SAMPLES_FACTORY_H
