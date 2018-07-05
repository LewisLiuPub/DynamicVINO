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

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "interactive_face_detection.hpp"
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include "cv_helpers.hpp"

#include <ext_list.hpp>

#include <opencv2/opencv.hpp>

#include <librealsense2/rs.hpp>


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

template<typename T>
void matU8ToBlob(const cv::Mat &orig_image, Blob::Ptr &blob, float scaleFactor = 1.0, int batchIndex = 0) {
    SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    T *blob_data = blob->buffer().as<T *>();

    cv::Mat resized_image(orig_image);
    if (width != orig_image.size().width || height != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + c * width * height + h * width + w] =
                        resized_image.at<cv::Vec3b>(h, w)[c] * scaleFactor;
            }
        }
    }
}

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
    ExecutableNetwork net;
    InferencePlugin *plugin;
    InferRequest::Ptr request;
    std::string &commandLineFlag;
    std::string topoName;
    const int maxBatch;

    BaseDetection(std::string &commandLineFlag, std::string topoName, int maxBatch)
            : commandLineFlag(commandLineFlag), topoName(topoName), maxBatch(maxBatch) {}

    virtual ~BaseDetection() {}

    ExecutableNetwork *operator->() {
        return &net;
    }

    virtual CNNNetwork read()  = 0;

    virtual void submitRequest() {
        if (!enabled() || request == nullptr) return;
        request->StartAsync();
    }

    virtual void wait() {
        if (!enabled() || !request) return;
        request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }

    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }

    void printPerformanceCounts() {
        if (!enabled()) {
            return;
        }
        slog::info << "Performance counts for " << topoName << slog::endl << slog::endl;
        ::printPerformanceCounts(request->GetPerformanceCounts(), std::cout, false);
    }
};

struct FaceDetectionClass : BaseDetection {
    std::string input;
    std::string output;
    int maxProposalCount;
    int objectSize;
    int enquedFrames = 0;
    float width = 0;
    float height = 0;
    bool resultsFetched = false;
    std::vector<std::string> labels;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        if (!enquedFrames) return;
        enquedFrames = 0;
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void enqueue(const cv::Mat &frame) {
        if (!enabled()) return;

        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        width = frame.cols;
        height = frame.rows;

        Blob::Ptr inputBlob = request->GetBlob(input);

        matU8ToBlob<uint8_t>(frame, inputBlob);

        enquedFrames = 1;
    }


    FaceDetectionClass() : BaseDetection(FLAGS_m, "Face Detection", 1) {}

    CNNNetwork read() override {
        slog::info << "Loading network files for Face Detection" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is set to  " << maxBatch << slog::endl;
        netReader.getNetwork().setBatchSize(maxBatch);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Read labels (if any)**/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";

        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Face Detection inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one input");
        }
        InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Face Detection outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Face Detection network should have only one output");
        }
        DataPtr &_output = outputInfo.begin()->second;
        output = outputInfo.begin()->first;

        const CNNLayerPtr outputLayer = netReader.getNetwork().getLayerByName(output.c_str());
        if (outputLayer->type != "DetectionOutput") {
            throw std::logic_error("Face Detection network output layer(" + outputLayer->name +
                                   ") should be DetectionOutput, but was " + outputLayer->type);
        }

        if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
            throw std::logic_error("Face Detection network output layer (" +
                                   output + ") should have num_classes integer attribute");
        }

        const int num_classes = outputLayer->GetParamAsInt("num_classes");
        if (labels.size() != num_classes) {
            if (labels.size() == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else
                labels.clear();
        }
        const SizeVector outputDims = _output->getTensorDesc().getDims();
        maxProposalCount = outputDims[2];
        objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
                                   std::to_string(outputDims.size()));
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        slog::info << "Loading Face Detection model to the " << FLAGS_d << " plugin" << slog::endl;
        input = inputInfo.begin()->first;
        return netReader.getNetwork();
    }

    void fetchResults() {
        if (!enabled()) return;
        results.clear();
        if (resultsFetched) return;
        resultsFetched = true;
        const float *detections = request->GetBlob(output)->buffer().as<float *>();

        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];
            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];
            if (r.confidence <= FLAGS_t) {
                continue;
            }

            r.location.x = detections[i * objectSize + 3] * width;
            r.location.y = detections[i * objectSize + 4] * height;
            r.location.width = detections[i * objectSize + 5] * width - r.location.x;
            r.location.height = detections[i * objectSize + 6] * height - r.location.y;

            if (image_id < 0) {
                break;
            }
            if (FLAGS_r) {
                std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                          << r.location.height << ")"
                          << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
            }

            results.push_back(r);
        }
    }
};

struct AgeGenderDetection : BaseDetection {
    std::string input;
    std::string outputAge;
    std::string outputGender;
    int enquedFaces = 0;

    AgeGenderDetection() : BaseDetection(FLAGS_m_ag, "Age Gender", FLAGS_n_ag) {}

    void submitRequest() override {
        if (!enquedFaces) return;
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }

    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch
                       << ") processed by Age Gender detector" << slog::endl;
            return;
        }
        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        Blob::Ptr inputBlob = request->GetBlob(input);

        matU8ToBlob<float>(face, inputBlob, 1.0f, enquedFaces);

        enquedFaces++;
    }

    struct Result {
        float age;
        float maleProb;
    };

    Result operator[](int idx) const {
        Blob::Ptr genderBlob = request->GetBlob(outputGender);
        Blob::Ptr ageBlob = request->GetBlob(outputAge);

        return {ageBlob->buffer().as<float *>()[idx] * 100,
                genderBlob->buffer().as<float *>()[idx * 2 + 1]};
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for AgeGender" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_ag);

        /** Set batch size to 16 **/
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Age Gender"
                   << slog::endl;


        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_ag) + ".bin";
        netReader.ReadWeights(binFileName);

        // -----------------------------------------------------------------------------------------------------

        /** Age Gender network should have one input two outputs **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Age Gender inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Age gender topology should have only one input");
        }
        InputInfo::Ptr &inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::FP32);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Age Gender outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 2) {
            throw std::logic_error("Age Gender network should have two output layers");
        }
        auto it = outputInfo.begin();
        DataPtr ageOutput = (it++)->second;
        DataPtr genderOutput = (it++)->second;

        // if gender output is convolution, it can be swapped with age
        if (genderOutput->getCreatorLayer().lock()->type == "Convolution") {
            std::swap(ageOutput, genderOutput);
        }

        if (ageOutput->getCreatorLayer().lock()->type != "Convolution") {
            throw std::logic_error("In Age Gender network, age layer (" + ageOutput->getCreatorLayer().lock()->name +
                                   ") should be a Convolution, but was: " + ageOutput->getCreatorLayer().lock()->type);
        }

        if (genderOutput->getCreatorLayer().lock()->type != "SoftMax") {
            throw std::logic_error(
                    "In Age Gender network, gender layer (" + genderOutput->getCreatorLayer().lock()->name +
                    ") should be a SoftMax, but was: " + genderOutput->getCreatorLayer().lock()->type);
        }
        slog::info << "Age layer: " << ageOutput->getCreatorLayer().lock()->name << slog::endl;
        slog::info << "Gender layer: " << genderOutput->getCreatorLayer().lock()->name << slog::endl;

        outputAge = ageOutput->name;
        outputGender = genderOutput->name;

        slog::info << "Loading Age Gender model to the " << FLAGS_d_ag << " plugin" << slog::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct HeadPoseDetection : BaseDetection {
    std::string input;
    std::string outputAngleR = "angle_r_fc";
    std::string outputAngleP = "angle_p_fc";
    std::string outputAngleY = "angle_y_fc";
    int enquedFaces = 0;
    cv::Mat cameraMatrix;

    HeadPoseDetection() : BaseDetection(FLAGS_m_hp, "Head Pose", FLAGS_n_hp) {}

    void submitRequest() override {
        if (!enquedFaces) return;
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }

    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch
                       << ") processed by Head Pose detector" << slog::endl;
            return;
        }
        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        Blob::Ptr inputBlob = request->GetBlob(input);

        matU8ToBlob<float>(face, inputBlob, 1.0f, enquedFaces);

        enquedFaces++;
    }

    struct Results {
        float angle_r;
        float angle_p;
        float angle_y;
    };

    Results operator[](int idx) const {
        Blob::Ptr angleR = request->GetBlob(outputAngleR);
        Blob::Ptr angleP = request->GetBlob(outputAngleP);
        Blob::Ptr angleY = request->GetBlob(outputAngleY);

        return {angleR->buffer().as<float *>()[idx],
                angleP->buffer().as<float *>()[idx],
                angleY->buffer().as<float *>()[idx]};
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for Head Pose detection " << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_hp);
        /** Set batch size to maximum currently set to one provided from command line **/
        netReader.getNetwork().setBatchSize(maxBatch);
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is sey to  " << netReader.getNetwork().getBatchSize() << " for Head Pose Network"
                   << slog::endl;
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_hp) + ".bin";
        netReader.ReadWeights(binFileName);


        /** Age Gender network should have one input two outputs **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Head Pose Network inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Head Pose topology should have only one input");
        }
        InputInfo::Ptr &inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::FP32);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Head Pose network outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 3) {
            throw std::logic_error("Head Pose network should have 3 outputs");
        }
        std::map<std::string, bool> layerNames = {
                {outputAngleR, false},
                {outputAngleP, false},
                {outputAngleY, false}
        };

        for (auto &&output : outputInfo) {
            CNNLayerPtr layer = output.second->getCreatorLayer().lock();
            if (layerNames.find(layer->name) == layerNames.end()) {
                throw std::logic_error("Head Pose network output layer unknown: " + layer->name + ", should be " +
                                       outputAngleR + " or " + outputAngleP + " or " + outputAngleY);
            }
            if (layer->type != "FullyConnected") {
                throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid type: " +
                                       layer->type + ", should be FullyConnected");
            }
            auto fc = dynamic_cast<FullyConnectedLayer *>(layer.get());
            if (fc->_out_num != 1) {
                throw std::logic_error("Head Pose network output layer (" + layer->name + ") has invalid out-size=" +
                                       std::to_string(fc->_out_num) + ", should be 1");
            }
            layerNames[layer->name] = true;
        }

        slog::info << "Loading Head Pose model to the " << FLAGS_d_hp << " plugin" << slog::endl;

        _enabled = true;
        return netReader.getNetwork();
    }

    void buildCameraMatrix(int cx, int cy, float focalLength) {
        if (!cameraMatrix.empty()) return;
        cameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
        cameraMatrix.at<float>(0) = focalLength;
        cameraMatrix.at<float>(2) = static_cast<float>(cx);
        cameraMatrix.at<float>(4) = focalLength;
        cameraMatrix.at<float>(5) = static_cast<float>(cy);
        cameraMatrix.at<float>(8) = 1;
    }

    void drawAxes(cv::Mat &frame, cv::Point3f cpoint, Results headPose, float scale) {
        double yaw = headPose.angle_y;
        double pitch = headPose.angle_p;
        double roll = headPose.angle_r;

        if (FLAGS_r) {
            std::cout << "Head pose results: yaw, pitch, roll = " << yaw << ";" << pitch << ";" << roll << std::endl;
        }

        pitch *= CV_PI / 180.0;
        yaw *= CV_PI / 180.0;
        roll *= CV_PI / 180.0;

        cv::Matx33f Rx(1, 0, 0,
                       0, cos(pitch), -sin(pitch),
                       0, sin(pitch), cos(pitch));
        cv::Matx33f Ry(cos(yaw), 0, -sin(yaw),
                       0, 1, 0,
                       sin(yaw), 0, cos(yaw));
        cv::Matx33f Rz(cos(roll), -sin(roll), 0,
                       sin(roll), cos(roll), 0,
                       0, 0, 1);


        auto r = cv::Mat(Rz * Ry * Rx);
        buildCameraMatrix(frame.cols / 2, frame.rows / 2, 950.0);

        cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

        xAxis.at<float>(0) = 1 * scale;
        xAxis.at<float>(1) = 0;
        xAxis.at<float>(2) = 0;

        yAxis.at<float>(0) = 0;
        yAxis.at<float>(1) = -1 * scale;
        yAxis.at<float>(2) = 0;

        zAxis.at<float>(0) = 0;
        zAxis.at<float>(1) = 0;
        zAxis.at<float>(2) = -1 * scale;

        zAxis1.at<float>(0) = 0;
        zAxis1.at<float>(1) = 0;
        zAxis1.at<float>(2) = 1 * scale;

        cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
        o.at<float>(2) = cameraMatrix.at<float>(0);

        xAxis = r * xAxis + o;
        yAxis = r * yAxis + o;
        zAxis = r * zAxis + o;
        zAxis1 = r * zAxis1 + o;

        cv::Point p1, p2;

        p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

        p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

        p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);

        p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
        p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
        cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);
        cv::circle(frame, p2, 3, cv::Scalar(255, 0, 0), 2);
    }
};

struct EmotionsDetectionClass : BaseDetection {
    std::string input;
    std::string outputEmotions;
    int enquedFaces = 0;

    EmotionsDetectionClass() : BaseDetection(FLAGS_m_em, "Emotions Recognition", FLAGS_n_em) {}

    void submitRequest() override {
        if (!enquedFaces) return;
        BaseDetection::submitRequest();
        enquedFaces = 0;
    }

    void enqueue(const cv::Mat &face) {
        if (!enabled()) {
            return;
        }
        if (enquedFaces == maxBatch) {
            slog::warn << "Number of detected faces more than maximum(" << maxBatch
                       << ") processed by Emotions detector" << slog::endl;
            return;
        }
        if (!request) {
            request = net.CreateInferRequestPtr();
        }

        Blob::Ptr inputBlob = request->GetBlob(input);

        matU8ToBlob<float>(face, inputBlob, 1.0f, enquedFaces);

        enquedFaces++;
    }

    std::string operator[](int idx) const {
        /* vector of supported emotions */
        static const std::vector<std::string> emotionsVec = {"neutral", "happy", "sad", "surprise", "anger"};
        auto emotionsVecSize = emotionsVec.size();

        Blob::Ptr emotionsBlob = request->GetBlob(outputEmotions);

        /* emotions vector must have the same size as number of channels
         * in model output. Default output format is NCHW so we check index 1. */
        int numOfChannels = emotionsBlob->getTensorDesc().getDims().at(1);
        if (numOfChannels != emotionsVec.size()) {
            throw std::logic_error("Output size (" + std::to_string(numOfChannels) +
                                   ") of the Emotions Recognition network is not equal "
                                   "to used emotions vector size (" +
                                   std::to_string(emotionsVec.size()) + ")");
        }

        auto emotionsValues = emotionsBlob->buffer().as<float *>();
        auto outputIdxPos = emotionsValues + idx;

        /* we identify an index of the most probable emotion in output array
           for idx image to return appropriate emotion name */
        int maxProbEmotionIx = std::max_element(outputIdxPos, outputIdxPos + emotionsVecSize) - outputIdxPos;
        return emotionsVec[maxProbEmotionIx];
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for Emotions recognition" << slog::endl;
        InferenceEngine::CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_em);

        /** Default batch size is 16 **/
        netReader.getNetwork().setBatchSize(maxBatch);
        slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize() << " for Emotions recognition"
                   << slog::endl;


        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_em) + ".bin";
        netReader.ReadWeights(binFileName);

        // -----------------------------------------------------------------------------------------------------

        /** Emotions recognition network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        slog::info << "Checking Emotions Recognition inputs" << slog::endl;
        InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Emotions Recognition topology should have only one input");
        }
        auto &inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::FP32);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        input = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        slog::info << "Checking Emotions Recognition outputs" << slog::endl;
        InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Emotions Recognition network should have one output layer");
        }

        DataPtr emotionsOutput = outputInfo.begin()->second;

        if (emotionsOutput->getCreatorLayer().lock()->type != "SoftMax") {
            throw std::logic_error("In Emotions Recognition network, Emotion layer ("
                                   + emotionsOutput->getCreatorLayer().lock()->name +
                                   ") should be a SoftMax, but was: " +
                                   emotionsOutput->getCreatorLayer().lock()->type);
        }
        slog::info << "Emotions layer: " << emotionsOutput->getCreatorLayer().lock()->name << slog::endl;

        outputEmotions = emotionsOutput->name;

        slog::info << "Loading Emotions Recognition model to the " << FLAGS_d_em << " plugin" << slog::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct Load {
    BaseDetection &detector;

    explicit Load(BaseDetection &detector) : detector(detector) {}

    void into(InferencePlugin &plg) const {
        if (detector.enabled()) {
            detector.net = plg.LoadNetwork(detector.read(), {});
            detector.plugin = &plg;
        }
    }
};

int main(int argc, char *argv[]) {
    try {
        /** This sample covers 3 certain topologies and cannot be generalized **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;

        cv::VideoCapture cap;
        // Declare RealSense pipeline, encapsulating the actual device and sensors
        rs2::pipeline pipe;
        //Create a configuration for configuring the pipeline with a non default profile
        rs2::config cfg;
        //Add desired streams to configuration
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

        size_t width = 0;
        size_t height = 0;
        const bool isRealSense = FLAGS_i == "realSense";
        const bool isCamera = FLAGS_i == "cam";

        if (isRealSense) {
            // Start streaming with default recommended configuration
            pipe.start(cfg);
        } else {
            if (!(isCamera ? cap.open(0) : cap.open(FLAGS_i))) {
                throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
            }
            width = (size_t) cap.get(CV_CAP_PROP_FRAME_WIDTH);
            height = (size_t) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        }


        // read input (video) frame
        cv::Mat frame;
        if (isRealSense) {
            // Camera warmup - dropping several first frames to let auto-exposure stabilize
            rs2::frameset frames;
            for(int i = 0; i < 30; i++)
            {
                //Wait for all configured streams to produce a frame
                frames = pipe.wait_for_frames();
            }

            width = 640;
            height = 480;
        } else if (!cap.read(frame)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        std::map<std::string, InferencePlugin> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
                {FLAGS_d,    FLAGS_m},
                {FLAGS_d_ag, FLAGS_m_ag},
                {FLAGS_d_hp, FLAGS_m_hp},
                {FLAGS_d_em, FLAGS_m_em}
        };

        FaceDetectionClass FaceDetection;
        AgeGenderDetection AgeGender;
        HeadPoseDetection HeadPose;
        EmotionsDetectionClass EmotionsDetection;

        for (auto &&option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName == "" || networkName == "") {
                continue;
            }

            if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                continue;
            }
            slog::info << "Loading plugin " << deviceName << slog::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            /** Load extensions for the CPU plugin **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<MKLDNNPlugin::IMKLDNNExtension>(FLAGS_l);
                    plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            pluginsForDevices[deviceName] = plugin;
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto &&plugin : pluginsForDevices) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR models and load them to plugins ------------------------------
        Load(FaceDetection).into(pluginsForDevices[FLAGS_d]);
        Load(AgeGender).into(pluginsForDevices[FLAGS_d_ag]);
        Load(HeadPose).into(pluginsForDevices[FLAGS_d_hp]);
        Load(EmotionsDetection).into(pluginsForDevices[FLAGS_d_em]);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Do inference ---------------------------------------------------------
        slog::info << "Start inference " << slog::endl;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto wallclock = std::chrono::high_resolution_clock::now();

        double ocv_decode_time = 0, ocv_render_time = 0;
        bool firstFrame = true;


        using namespace cv;
        const auto window_name = "Detection results";
        namedWindow(window_name, WINDOW_AUTOSIZE);

        /** Start inference & calc performance **/
        while (waitKey(1) < 0 && cvGetWindowHandle(window_name)) {
            if (isRealSense) {
                rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
                rs2::frame color_frame = data.get_color_frame();
                frame = Mat(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
            } else {
                /** requesting new frame if any*/
                cap.grab();
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            FaceDetection.enqueue(frame);
            auto t1 = std::chrono::high_resolution_clock::now();
            ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // ----------------------------Run face detection inference-----------------------------------------
            FaceDetection.submitRequest();
            FaceDetection.wait();

            t1 = std::chrono::high_resolution_clock::now();
            ms detection = std::chrono::duration_cast<ms>(t1 - t0);

            FaceDetection.fetchResults();

            for (auto &&face : FaceDetection.results) {
                if (AgeGender.enabled() || HeadPose.enabled() || EmotionsDetection.enabled()) {
                    auto clippedRect = face.location & cv::Rect(0, 0, width, height);
                    cv::Mat face = frame(clippedRect);
                    AgeGender.enqueue(face);
                    HeadPose.enqueue(face);
                    EmotionsDetection.enqueue(face);
                }
            }
            // ----------------------------Run age-gender, and head pose detection simultaneously---------------
            t0 = std::chrono::high_resolution_clock::now();
            if (AgeGender.enabled() || HeadPose.enabled() || EmotionsDetection.enabled()) {
                AgeGender.submitRequest();
                HeadPose.submitRequest();
                EmotionsDetection.submitRequest();

                AgeGender.wait();
                HeadPose.wait();
                EmotionsDetection.wait();
            }
            t1 = std::chrono::high_resolution_clock::now();
            ms secondDetection = std::chrono::duration_cast<ms>(t1 - t0);

            // ----------------------------Processing outputs---------------------------------------------------
            std::ostringstream out;
            out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                << (ocv_decode_time + ocv_render_time) << " ms";
            cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

            out.str("");
            out << "Face detection time: " << std::fixed << std::setprecision(2) << detection.count()
                << " ms ("
                << 1000.f / detection.count() << " fps)";
            cv::putText(frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                        cv::Scalar(255, 0, 0));

            if (HeadPose.enabled() || AgeGender.enabled() || EmotionsDetection.enabled()) {
                out.str("");
                out << (AgeGender.enabled() ? "Age Gender " : "")
                    << (AgeGender.enabled() && (HeadPose.enabled() || EmotionsDetection.enabled()) ? "+ " : "")
                    << (HeadPose.enabled() ? "Head Pose " : "")
                    << (HeadPose.enabled() && EmotionsDetection.enabled() ? "+ " : "")
                    << (EmotionsDetection.enabled() ? "Emotions Recognition " : "")
                    << "time: " << std::fixed << std::setprecision(2) << secondDetection.count()
                    << " ms ";
                if (!FaceDetection.results.empty()) {
                    out << "(" << 1000.f / secondDetection.count() << " fps)";
                }
                cv::putText(frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
            }

            int i = 0;
            for (auto &result : FaceDetection.results) {
                cv::Rect rect = result.location;

                out.str("");

                if (AgeGender.enabled() && i < AgeGender.maxBatch) {
                    out << (AgeGender[i].maleProb > 0.5 ? "M" : "F");
                    out << std::fixed << std::setprecision(0) << "," << AgeGender[i].age;
                    if (FLAGS_r) {
                        std::cout << "Predicted gender, age = " << out.str() << std::endl;
                    }
                } else {
                    out << (result.label < FaceDetection.labels.size() ? FaceDetection.labels[result.label] :
                            std::string("label #") + std::to_string(result.label))
                        << ": " << std::fixed << std::setprecision(3) << result.confidence;
                }

                if (EmotionsDetection.enabled()) {
                    /* currently we display only most probable emotion */
                    std::string emotion = EmotionsDetection[i];
                    if (FLAGS_r) {
                        std::cout << "Predicted emotion = " << emotion << std::endl;
                    }
                    out << "," << emotion;
                }

                cv::putText(frame,
                            out.str(),
                            cv::Point2f(result.location.x, result.location.y - 15),
                            cv::FONT_HERSHEY_COMPLEX_SMALL,
                            0.8,
                            cv::Scalar(0, 0, 255));

                if (HeadPose.enabled() && i < HeadPose.maxBatch) {
                    cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
                    HeadPose.drawAxes(frame, center, HeadPose[i], 50);
                }

                auto genderColor = (AgeGender.enabled() && (i < AgeGender.maxBatch)) ?
                                   ((AgeGender[i].maleProb < 0.5) ? cv::Scalar(147, 20, 255) : cv::Scalar(255, 0, 0)) :
                                   cv::Scalar(100, 100, 100);

                cv::rectangle(frame, result.location, genderColor, 1);

                i++;
            }

            if (-1 != cv::waitKey(1))
                break;

            t0 = std::chrono::high_resolution_clock::now();
            if (!FLAGS_no_show) {
                cv::imshow(window_name, frame);
            }
            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            // end of file, for single frame file, like image we just keep it displayed to let user check what was shown
            if (!isRealSense) {
                if (!cap.retrieve(frame)) {
                    if (!FLAGS_no_wait) {
                        slog::info << "Press any key to exit" << slog::endl;
                        cv::waitKey(0);
                    }
                    break;
                }
            }

            if (firstFrame) {
                slog::info << "Press any key to stop" << slog::endl;
            }

            firstFrame = false;
        }

        /** Show performace results **/
        if (FLAGS_pc) {
            FaceDetection.printPerformanceCounts();
            AgeGender.printPerformanceCounts();
            HeadPose.printPerformanceCounts();
        }
        // -----------------------------------------------------------------------------------------------------
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
