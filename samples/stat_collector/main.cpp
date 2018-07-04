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

#include <iostream>
#include <cfloat>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <gflags/gflags.h>

#include "utils.h"

#include "network_stats.h"
#include <ie_extension.h>
#include <ie_icnn_network_stats.hpp>

#include "data_stats.h"

#include <cnn_network_stats_impl.hpp>

#include <samples/slog.hpp>

using namespace InferenceEngine;
using InferenceEngine::details::CNNNetworkStatsImpl;
using InferenceEngine::details::CNNNetworkStatsImplPtr;
using InferenceEngine::details::InferenceEngineException;

static const char help_message[] = "Print a usage message.";
static const char stat_message[] = "Collect statistics to stat file";
static const char image_message[] = "Required. Path to an .bmp image or to validation dataset txt file";
static const char s_message[] = "Use statistics file";
static const char plugin_path_message[] = "Path to a plugin folder.";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
                                     "the sample will look for this plugin only";
static const char target_device_message[] = "Specify the target device to infer on; CPU or GPU is acceptable. " \
                                            "Sample will look for a suitable plugin for device specified";
static const char validation_message[] = "Optional. Path to a validation set .txt file.";
static const char validation_ref_message[] = "Optional. Validate both original and converted networks.";
static const char distance_message[] = "Optional. Calculate output distances.";
static const char batch_message[] = "Optional. Batch size.";
static const char target_stat_path_message[] = "Target statistics file path.";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for clDNN (GPU)-targeted custom kernels."
                                           "Absolute path to the xml file with the kernel descriptions";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers."
                                                 "Absolute path to a shared library with the kernel implementations";

#define DEFAULT_PATH_P "./lib"

DEFINE_bool(h, false, help_message);
DEFINE_bool(stat, false, stat_message);
DEFINE_string(i, "", image_message);
DEFINE_string(s, "", s_message);
DEFINE_string(m, "", model_message);
DEFINE_string(p, "", plugin_message);
DEFINE_string(pp, DEFAULT_PATH_P, plugin_path_message);
DEFINE_string(d, "", target_device_message);
DEFINE_bool(vr, false, validation_ref_message);
DEFINE_bool(dist, false, distance_message);
DEFINE_string(v, "", validation_message);
DEFINE_int32(b, 1, batch_message);
DEFINE_string(t, "", target_stat_path_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "stat_collector [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -v \"<path>\"             " << validation_message << std::endl;
    std::cout << "    -vr \"<path>\"            " << validation_ref_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -b \"<batch>\"            " << batch_message << std::endl;
    std::cout << "    -t \"<path>\"             " << target_stat_path_message << std::endl;
    std::cout << "    -dist                     " << distance_message << std::endl;
}

void CheckArgumentsAndExit() {
    if (FLAGS_h) {
        showUsage();
        exit(1);
    }

    bool noPluginAndBadDevice = FLAGS_p.empty() && FLAGS_d.compare("CPU") && FLAGS_d.compare("GPU");

    if (FLAGS_i.empty() || FLAGS_m.empty() || noPluginAndBadDevice) {
        if (noPluginAndBadDevice) std::cout << "ERROR: device is not supported" << std::endl;
        if (FLAGS_m.empty()) std::cout << "ERROR: file with model - not set" << std::endl;
        if (FLAGS_i.empty()) std::cout << "ERROR: image(s) for inference - not set" << std::endl;
        showUsage();
        exit(2);
    }
}

void GetInputImages(const std::string& path, std::vector<std::string>& images) {
    if (path.empty()) {
        return;
    }

    if (!IsDirectory(path)) {
        std::string ext = path.substr(path.find_last_of('.'));

        ext = FileExt(ToUpper(ext));

        if (ext == ".TXT") {
            std::string parentPath = FilePath(path);

            std::vector<size_t> classes;

            ParseValFile(path, images, classes);

            for (size_t i = 0; i < images.size(); i++) {
                images[i] = parentPath + images[i];
            }
        } else if (ext == ".BMP") {
            images.push_back(path);
        }
    } else {
        GetFilesInDir(path, ".BMP", images);
    }
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    CheckArgumentsAndExit();

    slog::info << "Loading plugin" << slog::endl;

    InferencePlugin plugin = PluginDispatcher({ FLAGS_pp, "../../../lib/intel64" , "" }).getPluginByDevice(FLAGS_d);

    if (!FLAGS_l.empty()) {
        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
        plugin.AddExtension(extension_ptr);
        slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
    }
    if (!FLAGS_c.empty()) {
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
    }

    NetworkStatsCollector netStats(plugin);

    size_t batchSize = FLAGS_b;

    std::string inLabelsPath = FileNameNoExt(FLAGS_m) + ".labels";

    std::string outStatsXmlPath = FLAGS_t/* + FileNameNoPath(FileNameNoExt(FLAGS_m)) + "-stats.xml"*/;
    std::string outStatsBinPath = FileNameNoExt(outStatsXmlPath) + ".bin";

    std::vector<std::string> images;
    GetInputImages(FLAGS_i, images);

    std::cout << "Collecting statistics for model " << FileNameNoPath(FLAGS_m) << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;

    std::string statsFile = FLAGS_s;

    if (images.empty()) {
        std::cerr << "No input images found" << std::endl;
        return 1;
    } else {
        std::cout << "Using " << images.size() << " images" << std::endl;
    }

    std::map<std::string, NetworkNodeStatsPtr> netNodesStats;
    try {
        netStats.LoadNetwork(FLAGS_m, batchSize);

        std::cout << "Inferencing and collecting statistics..." << std::endl;
        netStats.InferAndCollectStats(images, netNodesStats);

        CNNNetworkStatsImplPtr statsPtr(new CNNNetworkStatsImpl(netNodesStats));
        statsPtr->SaveToFile(outStatsXmlPath, outStatsBinPath);
    }
    catch (const InferenceEngineException& ex) {
        std::cerr << ex.what() << std::endl;
        return 3;
    }
    return 0;
}
