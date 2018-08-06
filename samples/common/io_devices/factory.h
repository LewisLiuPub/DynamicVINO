//
// Created by chris on 18-7-11.
//

#ifndef SAMPLES_FACTORY_H
#define SAMPLES_FACTORY_H

#include "input.h"

#include <memory>

#include <inference_engine.hpp>
#include <ext_list.hpp>

#include "mkldnn/mkldnn_extension_ptr.hpp"
#include "samples/common.hpp"

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
  static std::shared_ptr<BaseInputDevice> makeInputDeviceByName(const std::string &);
  static std::shared_ptr<InferenceEngine::InferencePlugin> makePluginByName(
      const std::string &, const std::string &, const std::string &, bool);
};

#endif //SAMPLES_FACTORY_H
