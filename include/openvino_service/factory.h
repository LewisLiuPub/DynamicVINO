//
// Created by chris on 18-7-11.
//

#ifndef SAMPLES_FACTORY_H
#define SAMPLES_FACTORY_H

#include "openvino_service/inputs/base_input.h"

#include <memory>

#include <inference_engine.hpp>
#include <extension/ext_list.hpp>

#include "mkldnn/mkldnn_extension_ptr.hpp"
#include "openvino_service/common.hpp"

/**
* @class Factory
* @brief This class is a factory class that produces the derived input device class corresponding to the input string
*/
class Factory {
 public:
  /**
  * @brief This function produces the derived input device class corresponding to the input string
  * @param[in] input device name, can be RealSenseCamera, StandardCamera or video directory
  * @return the instance of derived input device referenced by a smart pointer
  */
  static std::shared_ptr<Input::BaseInputDevice> makeInputDeviceByName(const std::string &);
  /**
  * @brief This function produces the derived inference plugin corresponding to the input string
  * @return the instance of derived inference plugin referenced by a smart pointer
  */
  static std::shared_ptr<InferenceEngine::InferencePlugin> makePluginByName(
      const std::string &, const std::string &, const std::string &, bool);
};

#endif //SAMPLES_FACTORY_H
