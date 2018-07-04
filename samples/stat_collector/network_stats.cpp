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

#include <cfloat>
#include <fstream>
#include <limits>
#include <memory>

#include <pugixml.hpp>
#include <cpp/ie_cnn_net_reader.h>

#include <format_reader_ptr.h>

#include "network_stats.h"
#include "data_stats.h"

#include <samples/slog.hpp>
#include "utils.h"
// #include "config.h"

using namespace InferenceEngine;

CNNLayerPtr addScaleShiftBeforeLayer(std::string name, CNNLayer::Ptr beforeLayer, size_t port, std::vector<float> scale) {
    if (beforeLayer->insData.size() < port) {
        THROW_IE_EXCEPTION << "cannot find appropraite port for addScaleShiftBeforeLayer";
    }

    DataPtr pData = beforeLayer->insData[port].lock();
    LayerParams params;
    params.name = name;
    params.precision = Precision::FP32;
    params.type = "ScaleShift";
    CNNLayerPtr lptr = std::make_shared<ScaleShiftLayer>(params);
    ScaleShiftLayer *pScaleShift = dynamic_cast<ScaleShiftLayer *>(lptr.get());

    SizeVector wdims({ pData->dims[2] });

    if (scale.size() == 1) {
        scale.resize(wdims[0]);
        for (int i = 1; i < wdims[0]; i++) {
            scale[i] = scale[0];
        }
    }

    if (scale.size() != pData->dims[2]) {
        THROW_IE_EXCEPTION << "Failed to add scaleshift before " << beforeLayer->name << " due to scales and layer output dims incossitency";
    }

    Blob::Ptr weights = nullptr;
    weights = make_shared_blob<float>(Precision::FP32, Layout::C, wdims);
    weights->allocate();
    float *buffer = weights->buffer().as<float *>();
    for (size_t i = 0, idx = 0; i < pData->dims[2]; i++) {
        buffer[i] = scale[i];
    }
    pScaleShift->_weights = weights;


    SizeVector bdims({ pData->dims[2] });
    Blob::Ptr biases = nullptr;
    biases = make_shared_blob<float>(Precision::FP32, Layout::C, bdims);
    biases->allocate();
    buffer = biases->buffer().as<float *>();
    for (size_t i = 0, idx = 0; i < pData->dims[2]; i++) {
        buffer[i] = 0.f;
    }
    pScaleShift->_biases = biases;

    Data *edge2 = new Data(*pData.get());
    DataPtr newEdge(edge2);
    lptr->insData.push_back(pData);
    lptr->outData.push_back(newEdge);
    newEdge->name = /*"EdgeAfter_" +*/ params.name;
    newEdge->creatorLayer = lptr;
    newEdge->inputTo.clear();
    newEdge->inputTo[beforeLayer->name] = beforeLayer;

    pData->inputTo.erase(beforeLayer->name);
    pData->inputTo[params.name] = lptr;

    for (size_t i = 0; i < beforeLayer->insData.size(); i++) {
        DataPtr d = beforeLayer->insData[i].lock();
        if (d == pData) {
            beforeLayer->insData[i] = newEdge;
            break;
        }
    }
    return lptr;
}

NetworkStatsCollector::NetworkStatsCollector(InferenceEnginePluginPtr plugin) : _plugin(plugin) {
}

NetworkStatsCollector::~NetworkStatsCollector() {
}

void NetworkStatsCollector::ReadNetworkAndSetWeights(const void *model, size_t size, const TBlob<uint8_t>::Ptr &weights, size_t batch) {
    /** Reading network model **/
    _networkReader.ReadNetwork(model, size);
    _networkReader.SetWeights(weights);
    auto network = _networkReader.getNetwork();
    network.setBatchSize(batch);
}

void NetworkStatsCollector::LoadNetwork(const std::string& modelPath, size_t batch) {
    /** Reading network model **/
    _networkReader.ReadNetwork(modelPath);
    _networkReader.ReadWeights(FileNameNoExt(modelPath) + ".bin");
    auto network = _networkReader.getNetwork();
    network.setBatchSize(batch);
}

void NetworkStatsCollector::InferAndCollectStats(const std::vector<std::string>& images,
                                                 std::map<std::string, NetworkNodeStatsPtr>& netNodesStats) {
    auto network = _networkReader.getNetwork();
    slog::info << "Collecting statistics for layers:" << slog::endl;

    std::vector<CNNLayerPtr> layersAfterInputs;

    std::string hackPrefix = "scaleshifted_input:";

    std::map<std::string, std::string> inputsFromLayers;
    for (auto&& layer : network) {
        if (layer->insData.size() > 0) {
            std::string inName = layer->input()->getName();
            for (auto&& input : network.getInputsInfo()) {
                if (inName == input.first) {
                    layersAfterInputs.push_back(layer);
                    inputsFromLayers[hackPrefix + layer->name] = inName;
                }
            }
        }
    }

    for (auto&& layer : layersAfterInputs) {
        std::string firstInputName = hackPrefix + layer->name;
        auto scaleShiftLayer = addScaleShiftBeforeLayer(firstInputName, layer, 0, { 1.f });
        ((ICNNNetwork&)network).addLayer(scaleShiftLayer);
    }

    // Adding output to every layer
    for (auto&& layer : network) {
        slog::info << "\t" << layer->name << slog::endl;

        std::string layerType = network.getLayerByName(layer->name.c_str())->type;
        if (layerType != "Split" && layerType != "Input") {
            network.addOutput(layer->name);
        }
    }

    NetworkNodeStatsPtr nodeStats;

    const size_t batchSize = network.getBatchSize();

    std::vector<std::string> imageNames;

    size_t rounded = images.size() - images.size() % batchSize;

    InferencePlugin plugin(_plugin);
    auto executable_network = plugin.LoadNetwork(network, {});

    std::map<std::string, std::vector<float>> min_outputs, max_outputs;

    for (size_t i = 0; i < rounded; i += batchSize) {
        imageNames.clear();

        for (size_t img = 0; img < batchSize; img++) {
            imageNames.push_back(images[i + img]);
        }


        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());

        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision of input data provided by the user.
         * This should be called before load of the network to the plugin **/
        inputInfoItem.second->setPrecision(Precision::FP32);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::shared_ptr<unsigned char>> imagesData;
        for (auto & i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(reader->getData(inputInfoItem.second->getDims()[0], inputInfoItem.second->getDims()[1]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        OutputsDataMap outputInfo(network.getOutputsInfo());
        for (auto itOut : outputInfo) {
            itOut.second->setPrecision(Precision::FP32);
        }

        auto infer_request = executable_network.CreateInferRequest();

        // -------------------------------Set input data----------------------------------------------------
        /** Iterate over all the input blobs **/

        /** Creating input blob **/
        Blob::Ptr input = infer_request.GetBlob(inputInfoItem.first);

        /** Filling input tensor with images. First b channel, then g and r channels **/
        size_t num_chanels = input->dims()[2];
        size_t image_size = input->dims()[1] * input->dims()[0];

        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

        /** Iterate over all input images **/
        for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t pid = 0; pid < image_size; pid++) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < num_chanels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in bytes            **/
                    data[image_id * image_size * num_chanels + ch * image_size + pid ] = imagesData.at(image_id).get()[pid*num_chanels + ch];
                }
            }
        }

        infer_request.Infer();


        for (auto itOut : outputInfo) {
            auto outBlob = infer_request.GetBlob(itOut.first);

            std::string outName = itOut.first;
            if (inputsFromLayers.find(itOut.first) != inputsFromLayers.end()) {
                outName = inputsFromLayers[itOut.first];
            }

            size_t N, C, statCount;
            if (outBlob->dims().size() == 4 && outBlob->layout() == Layout::NCHW) {
                N = outBlob->dims()[3];
                C = outBlob->dims()[2];
                statCount = C;
            } else if (outBlob->dims().size() == 2 && outBlob->layout() == Layout::NC) {
                N = outBlob->dims()[1];
                C = outBlob->dims()[0];
                statCount = 1;
            } else {
                slog::warn << "Only NCHW and NC layouts are supported. Skipping layer \"" << outName << "\"" << slog::endl;
                continue;
            }


            if (netNodesStats.find(outName) == netNodesStats.end()) {
                nodeStats = NetworkNodeStatsPtr(new NetworkNodeStats(statCount));

                netNodesStats[outName] = nodeStats;
            } else {
                nodeStats = netNodesStats[outName];
            }

            // Counting min/max outputs per channel
            for (size_t n = 0; n < N; n++) {
                if (outBlob->dims().size() == 4) {
                    size_t _HW = outBlob->dims()[0] * outBlob->dims()[1];
                    for (size_t c = 0; c < C; c++) {
                        if (outBlob->getTensorDesc().getPrecision() == Precision::FP32) {
                            float* ptr = &outBlob->buffer().as<float*>()[(n * C + c) * _HW];

                            float min = nodeStats->_minOutputs[c];
                            float max = nodeStats->_maxOutputs[c];
                            DataStats::GetDataMinMax<float>(ptr, _HW, min, max);
                            nodeStats->_minOutputs[c] = min;
                            nodeStats->_maxOutputs[c] = max;
                        } else if (outBlob->getTensorDesc().getPrecision() == Precision::U8) {
                            uint8_t* ptr = &outBlob->buffer().as<uint8_t*>()[(n * C + c) * _HW];

                            uint8_t min = nodeStats->_minOutputs[c];
                            uint8_t max = nodeStats->_maxOutputs[c];
                            DataStats::GetDataMinMax<uint8_t>(ptr, _HW, min, max);
                            nodeStats->_minOutputs[c] = min;
                            nodeStats->_maxOutputs[c] = max;
                        } else {
                            throw std::logic_error(std::string("Unsupported precision: ") + outBlob->getTensorDesc().getPrecision().name());
                        }
                    }
                } else if (outBlob->dims().size() == 2) {
                    if (outBlob->getTensorDesc().getPrecision() == Precision::FP32) {
                        float* ptr = &outBlob->buffer().as<float*>()[n * C];

                        float min = nodeStats->_minOutputs[0];
                        float max = nodeStats->_maxOutputs[0];
                        DataStats::GetDataMinMax<float>(ptr, C, min, max);
                        nodeStats->_minOutputs[0] = min;
                        nodeStats->_maxOutputs[0] = max;
                    } else if (outBlob->getTensorDesc().getPrecision() == Precision::U8) {
                        uint8_t* ptr = &outBlob->buffer().as<uint8_t*>()[n * C];

                        uint8_t min = nodeStats->_minOutputs[0];
                        uint8_t max = nodeStats->_maxOutputs[0];
                        DataStats::GetDataMinMax<uint8_t>(ptr, C, min, max);
                        nodeStats->_minOutputs[0] = min;
                        nodeStats->_maxOutputs[0] = max;
                    } else {
                        throw std::logic_error(std::string("Unsupported precision: ") + outBlob->getTensorDesc().getPrecision().name());
                    }
                }
            }
        }
    }
}