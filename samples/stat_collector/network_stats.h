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

#pragma once

#include <string>
#include <vector>
#include <map>

#include <inference_engine.hpp>

#include <ie_icnn_network_stats.hpp>

// #include "network.h"

class NetworkStatsCollector {
public:
    NetworkStatsCollector(InferenceEngine::InferenceEnginePluginPtr plugin);
    ~NetworkStatsCollector();

public:
    void ReadNetworkAndSetWeights(const void *model, size_t size, const InferenceEngine::TBlob<uint8_t>::Ptr &weights, size_t batch);
    void LoadNetwork(const std::string& modelPath, size_t batch);

    void InferAndCollectStats(const std::vector<std::string>& images,
                              std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>& netNodesStats);

/*    void InferAndCollectHistogram(const std::vector<std::string>& images,
                              const std::vector<std::string>& layerNames,
                              std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>& netNodesStats);

    void InferAndFindOptimalThreshold(const std::vector<std::string>& images,
                                  const std::vector<std::string>& layerNames,
                                  std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>& netNodesStats);

    void CalculateThreshold(std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>& netNodesStats);*/

    void CalculatePotentialMax(const float* weights, const InferenceEngine::SizeVector& weightDism, float& max);

private:
    InferenceEngine::CNNNetReader _networkReader;
    InferenceEngine::InferenceEnginePluginPtr _plugin;
};
