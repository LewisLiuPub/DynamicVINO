#include "openvino_service/factory.h"
#include "openvino_service/inputs/realsense_camera.h"
#include "openvino_service/inputs/standard_camera.h"
#include "openvino_service/inputs/video_input.h"

using namespace InferenceEngine;

std::shared_ptr<Input::BaseInputDevice>
Factory::makeInputDeviceByName(const std::string &input_device_name) {
  if (input_device_name == "RealSenseCamera") {
    return std::make_shared<Input::RealSenseCamera>();
  } else if (input_device_name == "StandardCamera") {
    return std::make_shared<Input::StandardCamera>();
  } else {
    return std::make_shared<Input::Video>(input_device_name);
  }
}

std::shared_ptr<InferencePlugin>
Factory::makePluginByName(const std::string &device_name,
                          const std::string &custom_cpu_library_message, //FLAGS_l
                          const std::string &custom_cldnn_message, //FLAGS_c
                          bool performance_message) { //FLAGS_pc
  InferencePlugin plugin =
      PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(
          device_name);
  /** Printing plugin version **/
  printPluginVersion(plugin, std::cout);
  /** Load extensions for the CPU plugin **/
  if ((device_name.find("CPU") != std::string::npos)) {
    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    if (!custom_cpu_library_message.empty()) {
      // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
      auto extension_ptr = make_so_pointer<MKLDNNPlugin::IMKLDNNExtension>(
          custom_cpu_library_message);
      plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
    }
  } else if (!custom_cldnn_message.empty()) {
    // Load Extensions for other plugins not CPU
    plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE,
                       custom_cldnn_message}});
  }
  if (performance_message) {
    plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT,
                       PluginConfigParams::YES}});
  }
  return std::make_shared<InferencePlugin>(
      InferenceEngine::InferenceEnginePluginPtr(plugin));
}