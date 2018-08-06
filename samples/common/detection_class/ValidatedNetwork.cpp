//
// Created by chris on 18-8-2.
//

#include "ValidatedNetwork.h"

//Validated Base Network
ValidatedBaseNetwork::ValidatedBaseNetwork(
    const std::string &model_loc, const std::string &device,
    size_t max_batch_size, int input_num, int output_num) :
    model_loc_(model_loc), device_(device),
      max_batch_size_(max_batch_size) {
  networkInit(net_reader_);
  checkNetworkSize(input_num, output_num, net_reader_);
}

void ValidatedBaseNetwork::networkInit(
    InferenceEngine::CNNNetReader::Ptr net_reader) {
    slog::info << "Loading network files" << slog::endl;
    //Read network model
    net_reader->ReadNetwork(model_loc_);
    //Set batch size to given max_batch_size_
    slog::info << "Batch size is set to  " << max_batch_size_ << slog::endl;
    net_reader->getNetwork().setBatchSize(max_batch_size_);
    //Extract model name and load it's weights
    std::string bin_file_name = fileNameNoExt(model_loc_) + ".bin";
    net_reader->ReadWeights(bin_file_name);
    //Read labels (if any)
    std::string label_file_name = fileNameNoExt(model_loc_) + ".labels";
    std::ifstream inputFile(label_file_name);
    std::copy(std::istream_iterator<std::string>(inputFile),
              std::istream_iterator<std::string>(),
              std::back_inserter(labels_));
  checkLayerProperty(net_reader);
    setLayerProperty(net_reader);
}

void ValidatedBaseNetwork::checkNetworkSize(
    int input_size,
    int output_size,
    InferenceEngine::CNNNetReader::Ptr net_reader) {
  //check input size
  slog::info << "Checking " + getNetworkName() + " inputs" << slog::endl;
  InferenceEngine::InputsDataMap
      input_info(net_reader->getNetwork().getInputsInfo());
  if (input_info.size() != input_size) {
    throw std::logic_error(
        "Face Detection network should have only one input");
  }
  //check output size
  slog::info << "Checking Face Detection outputs" << slog::endl;
  InferenceEngine::OutputsDataMap
      output_info(net_reader->getNetwork().getOutputsInfo());
  if (output_info.size() != output_size) {
    throw std::logic_error(
        "Face Detection network should have only one output");
  }
  InferenceEngine::DataPtr &output_data_ptr = output_info.begin()->second;
  input_ = input_info.begin()->first;
  output_ = output_info.begin()->first;
}

//Validated Face Detection Network
ValidatedFaceDetectionNetwork::ValidatedFaceDetectionNetwork(
    const std::string &model_loc, const std::string &device,
    size_t max_batch_size, int input_num, int output_num)
    : ValidatedBaseNetwork(model_loc, device, max_batch_size,
                           input_num, output_num){};

void ValidatedFaceDetectionNetwork::setLayerProperty(
    InferenceEngine::CNNNetReader::Ptr net_reader) {
  //set input property
  InferenceEngine::InputsDataMap
      input_info_map(net_reader->getNetwork().getInputsInfo());
  InferenceEngine::InputInfo::Ptr input_info = input_info_map.begin()->second;
  input_info->setPrecision(InferenceEngine::Precision::U8);
  input_info->setLayout(InferenceEngine::Layout::NCHW);
  //set output property
  InferenceEngine::OutputsDataMap
      output_info_map(net_reader->getNetwork().getOutputsInfo());
  InferenceEngine::DataPtr &output_data_ptr = output_info_map.begin()->second;
  output_data_ptr->setPrecision(InferenceEngine::Precision::FP32);
  output_data_ptr->setLayout(InferenceEngine::Layout::NCHW);
}

void ValidatedFaceDetectionNetwork::checkLayerProperty(
    const InferenceEngine::CNNNetReader::Ptr &net_reader) {
  slog::info << "Checking Face Detection outputs" << slog::endl;
  const InferenceEngine::CNNLayerPtr
      output_layer = net_reader->getNetwork().getLayerByName(
      getOutputLayerName().c_str());
  //output layer should be DetectionOutput type
  if (output_layer->type != "DetectionOutput") {
    throw std::logic_error(
        "Face Detection network output layer(" + output_layer->name +
            ") should be DetectionOutput, but was " + output_layer->type);
  }
  //output layer should have attribute called num_classes
  if (output_layer->params.find("num_classes") ==
      output_layer->params.end()) {
    throw std::logic_error("Face Detection network output layer (" +
        getOutputLayerName()+
        ") should have num_classes integer attribute");
  }
  //class number should be equal to size of label vector
  //if network has default "background" class, fake is used
  const int num_classes = output_layer->GetParamAsInt("num_classes");
  if (getLabels().size() != num_classes) {
    if (getLabels().size() == (num_classes - 1)) {
      getLabels().insert(getLabels().begin(), "fake");
    } else {
      getLabels().clear();
    }
  }
  //last dimension of output layer should be 7
  InferenceEngine::OutputsDataMap
      output_info_map(net_reader->getNetwork().getOutputsInfo());
  InferenceEngine::DataPtr &output_data_ptr = output_info_map.begin()->second;
  const InferenceEngine::SizeVector output_dims
      = output_data_ptr->getTensorDesc().getDims();
  max_proposal_count_ = static_cast<int>(output_dims[2]);
  object_size_= static_cast<int>(output_dims[3]);
  if (object_size_ != 7) {
    throw std::logic_error(
        "Face Detection network output layer should have 7 as a last dimension");
  }
  if (output_dims.size() != 4) {
    throw std::logic_error(
        "Face Detection network output dimensions not compatible shoulld be 4, "
        "but was " + std::to_string(output_dims.size()));
  }
}

const std::string ValidatedFaceDetectionNetwork::getNetworkName() const {
  return "Face Detection";
}