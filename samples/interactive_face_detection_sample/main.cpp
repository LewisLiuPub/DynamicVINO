/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine interactive_face_detection sample application
* \file interactive_face_detection_sample/main.cpp
* \example interactive_face_detection_sample/main.cpp
*/
#include "detection.h"
#include "pipeline.h"
#include "engine.h"
#include "interactive_face_detection.hpp"
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include "inference_engine.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"
#include "io_devices/factory.h"
#include "ext_list.hpp"
#include "opencv2/opencv.hpp"
#include "librealsense2/rs.hpp"

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

using namespace InferenceEngine;
using namespace rs2;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
  // ---------------------------Parsing and validation of input args--------------------------------------
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_h) {
    showUsage();
    return false;
  }
  slog::info << "Parsing input parameters" << slog::endl;

  if (FLAGS_i.empty()) {
    throw std::logic_error("Parameter -i is not set");
  }

  if (FLAGS_m.empty()) {
    throw std::logic_error("Parameter -m is not set");
  }

  if (FLAGS_n_ag < 1) {
    throw std::logic_error("Parameter -n_ag cannot be 0");
  }

  if (FLAGS_n_hp < 1) {
    throw std::logic_error("Parameter -n_hp cannot be 0");
  }

  return true;
}

int main(int argc, char *argv[]) {
  try {
    /** This sample covers 3 certain topologies and cannot be generalized **/
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion()
              << std::endl;

    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
      return 0;
    }

    slog::info << "Reading input" << slog::endl;

    std::shared_ptr<BaseInputDevice>
        input_device = Factory::makeInputDeviceByName(FLAGS_i);
    if (!input_device->initialize(0)) {
      throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
    }
    Pipeline pipe;
    pipe.add("", "video_input", input_device);
    cv::Mat frame;
    // --------------------------- 1. Load Plugin for inference engine -------------------------------------
    std::map<std::string, InferencePlugin> pluginsForDevices;
    std::vector<std::pair<std::string, std::string>> cmdOptions = {
        {FLAGS_d, FLAGS_m},
        {FLAGS_d_ag, FLAGS_m_ag},
        {FLAGS_d_hp, FLAGS_m_hp},
        {FLAGS_d_em, FLAGS_m_em}
    };
    for (auto &&option : cmdOptions) {
      auto deviceName = option.first;
      auto networkName = option.second;

      if (deviceName.empty() || networkName.empty()) {
        continue;
      }

      if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
        continue;
      }
      pluginsForDevices[deviceName] = *Factory::makePluginByName(
          deviceName, FLAGS_l, FLAGS_c, FLAGS_pc);
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 2. Read IR models and load them to plugins ------------------------------
    //generate face detection object
    auto face_detection_network =
        std::make_shared<ValidatedFaceDetectionNetwork>(
            FLAGS_m, FLAGS_d, 1, 1, 1);
    face_detection_network->networkInit();
    auto face_detection_engine =
        std::make_shared<NetworkEngine>(
            &pluginsForDevices[FLAGS_d],*face_detection_network);
    openvino_service::FaceDetection face_detection(
        face_detection_network->getMaxProposalCount(),
        face_detection_network->getObjectSize(), FLAGS_t);
    face_detection.loadNetwork(face_detection_network);
    face_detection.loadEngine(face_detection_engine);

    //generate emotions detection object
    auto emotions_detection_network =
        std::make_shared<ValidatedEmotionsClassificationNetwork>(
            FLAGS_m_em, FLAGS_d_em, 1, 1, 16);
    emotions_detection_network->networkInit();
    auto emotions_detection_engine =
        std::make_shared<NetworkEngine>(
            &pluginsForDevices[FLAGS_d_em],*emotions_detection_network);
    openvino_service::EmotionsDetection emotions_detection;
    emotions_detection.loadNetwork(emotions_detection_network);
    emotions_detection.loadEngine(emotions_detection_engine);

    //generate agegender detection object
    auto agegender_detection_network =
        std::make_shared<ValidatedAgeGenderNetwork>(
            FLAGS_m_ag, FLAGS_d_ag, 1, 2, 16);
    agegender_detection_network->networkInit();
    auto agegender_detection_engine =
        std::make_shared<NetworkEngine>(
            &pluginsForDevices[FLAGS_d_ag],*agegender_detection_network);
    openvino_service::AgeGenderDetection agegender_detection;
    agegender_detection.loadNetwork(agegender_detection_network);
    agegender_detection.loadEngine(agegender_detection_engine);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Test Pipeline ---------------------------------------------------------
    std::shared_ptr<openvino_service::BaseInference>
        face_detection_ptr(&face_detection);
    std::shared_ptr<openvino_service::BaseInference>
        emotions_detection_ptr(&emotions_detection);
    std::shared_ptr<openvino_service::BaseInference>
        agegender_detection_ptr(&agegender_detection);
    pipe.add("video_input", "face_detection", face_detection_ptr);
    pipe.add("face_detection", "agegender", agegender_detection_ptr);

    std::string window_name = "Detection results";
    std::shared_ptr<BaseOutput> output_ptr(new ImageWindow(window_name));
    pipe.add("agegender", "video_output", output_ptr);

    using namespace cv;
    pipe.setcallback();
    while (waitKey(1) < 0 && cvGetWindowHandle(window_name.c_str())) {
      pipe.runOnce();
    }
  }
  catch (const std::exception &error) {
    slog::err << error.what() << slog::endl;
    return 1;
  }
  catch (...) {
    slog::err << "Unknown/internal exception happened." << slog::endl;
    return 1;
  }

  slog::info << "Execution successful" << slog::endl;
  return 0;
}
