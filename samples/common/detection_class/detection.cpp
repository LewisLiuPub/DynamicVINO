//
// Created by chris on 18-7-12.
//

#include "detection.h"

using namespace InferenceEngine;
using namespace DetectionClass;
//utils

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

//Detection

Detection::Detection(const std::string &model_loc, const std::string &device, int max_batch_size) : model_loc_(model_loc), device_(device), max_batch_size_(max_batch_size) {}

CNNNetwork Detection::Read() {
    try {
        CNNNetReader net_reader;
        networkInit(&net_reader);
        initAndCheckInput(&net_reader);
        initAndCheckOutput(&net_reader);
        return net_reader.getNetwork();
    }
    catch (...) {
        throw std::runtime_error("read CNNNetwork error!");
    }
}
bool Detection::enabled() const {
    if (!enablingChecked_) {
        enable_ = !model_loc_.empty() && !device_.empty();
        if (!enable_) {
            slog::info << getName() << " DISABLED" << slog::endl;
        }
        enablingChecked_ = true;
    }
    return enable_;
}

void Detection::wait() {
    if (!enabled() || !request_) return;
    request_->Wait(IInferRequest::WaitMode::RESULT_READY);
}

void Detection::submitRequest() {
    if (!enabled() || request_ == nullptr) return;
    request_->StartAsync();
}

void Detection::load(InferenceEngine::InferencePlugin &plg) {
    if (enabled()) {
        network_ = plg.LoadNetwork(Read(), {});
        plugin_ = &plg;
    }
}

void Detection::printPerformanceCounts() {
    if (!enabled()) {
        return;
    }
    slog::info << "Performance counts for " << getName() << slog::endl << slog::endl;
    ::printPerformanceCounts(getRequest()->GetPerformanceCounts(), std::cout, false);
}

//FaceDetection

FaceDetection::FaceDetection(const std::string &model_loc,
                             const std::string &device, double show_output_thresh) : Detection(model_loc, device, 1), show_output_thresh_(show_output_thresh){};

void FaceDetection::enqueue(const cv::Mat &frame) {
    if (!enabled()) { return; }
    if (!getRequest()) {
        setRequest(getNetwork().CreateInferRequestPtr());
    }
    width_ = frame.cols;
    height_ = frame.rows;
    Blob::Ptr input_blob = getRequest()->GetBlob(input_);
    matU8ToBlob<uint8_t>(frame, input_blob);
    enqueued_frames = 1;
}

void FaceDetection::submitRequest() {
    if (!enqueued_frames) return;
    enqueued_frames = 0;
    results_fetched_ = false;
    results_.clear();
    Detection::submitRequest();
};

void FaceDetection::networkInit(CNNNetReader *net_reader) {
    slog::info << "Loading network files for Face Detection" << slog::endl;
    //Read network model
    net_reader->ReadNetwork(getModelLoc());
    //Set batch size to 1
    slog::info << "Batch size is set to  " << getMaxBatchSize() << slog::endl;
    net_reader->getNetwork().setBatchSize(getMaxBatchSize());
    //Extract model name and load it's weights
    std::string bin_file_name = fileNameNoExt(getModelLoc()) + ".bin";
    net_reader->ReadWeights(bin_file_name);
    //Read labels (if any)
    std::string label_file_name = fileNameNoExt(getModelLoc()) + ".labels";
    std::ifstream inputFile(label_file_name);
    std::copy(std::istream_iterator<std::string>(inputFile),
              std::istream_iterator<std::string>(),
              std::back_inserter(getLabels()));
}

void FaceDetection::initAndCheckInput(CNNNetReader *net_reader) {
    slog::info << "Checking Face Detection inputs" << slog::endl;
    InputsDataMap input_info(net_reader->getNetwork().getInputsInfo());
    if (input_info.size() != 1) {
        throw std::logic_error("Face Detection network should have only one input");
    }
    InputInfo::Ptr input_info_first = input_info.begin()->second;
    input_info_first->setPrecision(Precision::U8);
    input_info_first->setLayout(Layout::NCHW);
    input_ = input_info.begin()->first;
}

void FaceDetection::initAndCheckOutput(CNNNetReader *net_reader) {
    slog::info << "Checking Face Detection outputs" << slog::endl;
    OutputsDataMap output_info(net_reader->getNetwork().getOutputsInfo());
    if (output_info.size() != 1) {
        throw std::logic_error("Face Detection network should have only one output");
    }
    DataPtr & output_data_ptr = output_info.begin()->second;
    output_ = output_info.begin()->first;
    const CNNLayerPtr output_layer = net_reader->getNetwork().getLayerByName(output_.c_str());
    if (output_layer->type != "DetectionOutput") {
        throw std::logic_error("Face Detection network output layer(" + output_layer->name +
                               ") should be DetectionOutput, but was " + output_layer->type);
    }
    if (output_layer->params.find("num_classes") == output_layer->params.end()) {
        throw std::logic_error("Face Detection network output layer (" +
                               output_ + ") should have num_classes integer attribute");
    }
    const int num_classes = output_layer->GetParamAsInt("num_classes");
    if (getLabels().size() != num_classes) {
        if (getLabels().size() == (num_classes - 1))  // if network assumes default "background" class, having no label
            getLabels().insert(getLabels().begin(), "fake");
        else
            getLabels().clear();
    }
    const SizeVector output_dims = output_data_ptr->getTensorDesc().getDims();
    max_proposal_count_ = output_dims[2];
    object_size_ = output_dims[3];
    if (object_size_ != 7) {
        throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
    }
    if (output_dims.size() != 4) {
        throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
                               std::to_string(output_dims.size()));
    }
    output_data_ptr->setPrecision(Precision::FP32);
    output_data_ptr->setLayout(Layout::NCHW);
    slog::info << "Loading Face Detection model to the " << getDevice() << " plugin" << slog::endl;
}

void FaceDetection::fetchResults() {
    if (!enabled()) return;
    results_.clear();
    if (results_fetched_) return;
    results_fetched_ = true;
    const float *detections = getRequest()->GetBlob(output_)->buffer().as<float *>();

    for (int i = 0; i < max_proposal_count_; i++) {
        float image_id = detections[i * object_size_ + 0];
        Result r;
        r.label = static_cast<int>(detections[i * object_size_ + 1]);
        r.confidence = detections[i * object_size_ + 2];
        if (r.confidence <= show_output_thresh_) {
            continue;
        }
        r.location.x = static_cast<int>(detections[i * object_size_ + 3] * width_);
        r.location.y = static_cast<int>(detections[i * object_size_ + 4] * height_);
        r.location.width = static_cast<int>(detections[i * object_size_ + 5] * width_ - r.location.x);
        r.location.height = static_cast<int>(detections[i * object_size_ + 6] * height_ - r.location.y);

        if (image_id < 0) {
            break;
        }
        if (getRawOutput()) {
            std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                      "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                      << r.location.height << ")"
                      << ((r.confidence > show_output_thresh_) ? " WILL BE RENDERED!" : "") << std::endl;
        }

        results_.push_back(r);
    }
}