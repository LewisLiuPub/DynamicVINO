/**
 * @brief a header file with declaration of Factory class
 * @file factory.hpp
 */

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
  static std::unique_ptr<Input::BaseInputDevice>
      makeInputDeviceByName(const std::string &input_device_name);
  /**
  * @brief This function produces the derived inference plugin corresponding to the input string
  * @param[in] device_name The name of target device (CPU, GPU, FPGA, MYRIAD)
  * @param[in] custom_cpu_library_message Absolute path to CPU library with user layers
  * @param[in] custom_cldnn_message  clDNN custom kernels path
  * @param[in] performance_message Enable per-layer performance report
  * @return the instance of derived inference plugin referenced by a smart pointer
  */
  static std::unique_ptr<InferenceEngine::InferencePlugin> makePluginByName(
      const std::string &device_name,
      const std::string &custom_cpu_library_message,
      const std::string &custom_cldnn_message,
      bool performance_message);
};

#endif //SAMPLES_FACTORY_H
