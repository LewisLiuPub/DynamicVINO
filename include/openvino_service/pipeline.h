//
// Created by chris on 18-7-19.
//
/**
 * @brief a header file with definition of Pipeline class
 * @file pipeline.h
 */
#ifndef SAMPLES_PIPELINE_H
#define SAMPLES_PIPELINE_H

#include "openvino_service/inferences/base_inference.h"
#include "openvino_service/inputs/standard_camera.h"
#include "openvino_service/outputs/base_output.h"

#include "opencv2/opencv.hpp"

#include <memory>
#include <atomic>
#include <mutex>
#include <future>

/**
 * @class Pipeline
 * @brief This class is a pipeline class that stores the topology of 
 * the input device, output device and networks and make inference.
 */
class Pipeline {
 public:
  Pipeline();
  /**
   * @brief Add input device to the pipeline.
   * @param[in] name name of the current input device.
   * @param[in] input_device the input device instance to be added.
   * @return whether the add operation is successful
   */
  bool add(const std::string &name,
           std::shared_ptr<Input::BaseInputDevice> input_device);
  /**
   * @brief Add inference network to the pipeline.
   * @param[in] parent name of the parent device or inference.
   * @param[in] name name of the current inference network.
   * @param[in] inference the inference instance to be added.
   * @return whether the add operation is successful
   */
  bool add(const std::string &parent, const std::string &name,
           std::shared_ptr<openvino_service::BaseInference> inference);
  /**
   * @brief Add output device to the pipeline.
   * @param[in] parent name of the parent inference.
   * @param[in] name name of the current output device.
   * @param[in] output the output instance to be added.
   * @return whether the add operation is successful
   */         
  bool add(const std::string &parent, const std::string &name,
           std::shared_ptr<Outputs::BaseOutput> output);
  /**
   * @brief Add inference network-output device edge to the pipeline.
   * @param[in] parent name of the parent inference.
   * @param[in] name name of the current output device.
   * @return whether the add operation is successful
   */              
  bool add(const std::string &parent, const std::string &name);
  /**
   * @brief Do the inference once. 
   * Data flow from input device to inference network, then to output device.
   */
  void runOnce();
  /**
   * @brief The callback function provided for all the inference network in the pipeline.
   */
  void callback(const std::string &detection_name);
  /**
   * @brief Set the inference network to call the callback function as soon as each inference is finished.
   */
  void setcallback();
  void printPipeline();
 private:
  std::shared_ptr<Input::BaseInputDevice> input_device_;
  std::string input_device_name_;
  std::multimap<std::string, std::string> next_;
  std::map<std::string, std::shared_ptr<openvino_service::BaseInference>>
      name_to_detection_map_;
  std::map<std::string, std::shared_ptr<Outputs::BaseOutput>> name_to_output_map_;
  int total_inference_ = 0;
  std::set<std::string> output_names_;
  int width_ = 0;
  int height_ = 0;
  cv::Mat frame_;
  // for multi threads
  std::atomic<int> counter_;
  std::mutex counter_mutex_;
  std::condition_variable cv_;
};

#endif //SAMPLES_PIPELINE_H
