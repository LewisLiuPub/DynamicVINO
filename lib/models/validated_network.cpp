//
// Created by chris on 18-8-2.
//

#include "validated_network.h"

//Validated Base Network
ValidatedBaseNetwork::ValidatedBaseNetwork(
    const std::string &model_loc, const std::string &device,
    int input_num, int output_num, int max_batch_size) :
    input_num_(input_num), output_num_(output_num),
    model_loc_(model_loc), device_(device), max_batch_size_(max_batch_size) {
  if (model_loc.empty() || device.empty()) {
    throw std::logic_error("model file or device name is empty!");
  }
  net_reader_ = std::make_shared<InferenceEngine::CNNNetReader>();
}

void ValidatedBaseNetwork::networkInit() {
  slog::info << "Loading network files" << slog::endl;
  //Read network model
  net_reader_->ReadNetwork(model_loc_);
  //Set batch size to given max_batch_size_
  slog::info << "Batch size is set to  " << max_batch_size_ << slog::endl;
  net_reader_->getNetwork().setBatchSize(max_batch_size_);
  //Extract model name and load it's weights
  std::string bin_file_name = fileNameNoExt(model_loc_) + ".bin";
  net_reader_->ReadWeights(bin_file_name);
  //Read labels (if any)
  std::string label_file_name = fileNameNoExt(model_loc_) + ".labels";
  std::ifstream inputFile(label_file_name);
  std::copy(std::istream_iterator<std::string>(inputFile),
            std::istream_iterator<std::string>(),
            std::back_inserter(labels_));
  checkNetworkSize(input_num_, output_num_, net_reader_);
  checkLayerProperty(net_reader_);
  setLayerProperty(net_reader_);
}

void ValidatedBaseNetwork::checkNetworkSize(
    int input_size,
    int output_size,
    InferenceEngine::CNNNetReader::Ptr net_reader) {
  //check input size
  slog::info << "Checking input size" << slog::endl;
  InferenceEngine::InputsDataMap
      input_info(net_reader->getNetwork().getInputsInfo());
  if (input_info.size() != input_size) {
    throw std::logic_error(
        getNetworkName() + " should have only one input");
  }
  //check output size
  slog::info << "Checking output size" << slog::endl;
  InferenceEngine::OutputsDataMap
      output_info(net_reader->getNetwork().getOutputsInfo());
  if (output_info.size() != output_size) {
    throw std::logic_error(
        getNetworkName() + "network should have only one output");
  }
  InferenceEngine::DataPtr &output_data_ptr = output_info.begin()->second;
}

//Validated Face Detection Network
ValidatedFaceDetectionNetwork::ValidatedFaceDetectionNetwork(
    const std::string &model_loc, const std::string &device,
    int input_num, int output_num, int max_batch_size)
    : ValidatedBaseNetwork(model_loc, device,
                           input_num, output_num, max_batch_size){};

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
  //set input and output layer name
  input_ = input_info_map.begin()->first;
  output_ = output_info_map.begin()->first;
}

void ValidatedFaceDetectionNetwork::checkLayerProperty(
    const InferenceEngine::CNNNetReader::Ptr &net_reader) {
  slog::info << "Checking Face Detection outputs" << slog::endl;
  InferenceEngine::OutputsDataMap
      output_info_map(net_reader->getNetwork().getOutputsInfo());
  InferenceEngine::DataPtr &output_data_ptr = output_info_map.begin()->second;
  output_ = output_info_map.begin()->first;
  const InferenceEngine::CNNLayerPtr
      output_layer = net_reader->getNetwork().getLayerByName(
      output_.c_str());
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
        output_+
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
  const InferenceEngine::SizeVector output_dims
      = output_data_ptr->getTensorDesc().getDims();
  max_proposal_count_ = static_cast<int>(output_dims[2]);
  slog::info << "max proposal count is: " << max_proposal_count_ <<slog::endl;
  object_size_= static_cast<int>(output_dims[3]);
  if (object_size_ != 7) {
    throw std::logic_error(
        "Face Detection network output layer should have 7 as a last dimension"
        );
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

//Validated Emotions Detection Network
ValidatedEmotionsClassificationNetwork::ValidatedEmotionsClassificationNetwork(
    const std::string &model_loc, const std::string &device,
    int input_num, int output_num, int max_batch_size)
    : ValidatedBaseNetwork(model_loc, device,
                           input_num, output_num, max_batch_size){};

void ValidatedEmotionsClassificationNetwork::setLayerProperty(
    InferenceEngine::CNNNetReader::Ptr net_reader) {
  //set input property
  InferenceEngine::InputsDataMap
      input_info_map(net_reader->getNetwork().getInputsInfo());
  InferenceEngine::InputInfo::Ptr input_info = input_info_map.begin()->second;
  input_info->setPrecision(InferenceEngine::Precision::FP32);
  input_info->setLayout(InferenceEngine::Layout::NCHW);
  //set output property
  InferenceEngine::OutputsDataMap
      output_info_map(net_reader->getNetwork().getOutputsInfo());
  InferenceEngine::DataPtr &output_data_ptr = output_info_map.begin()->second;
  output_data_ptr->setPrecision(InferenceEngine::Precision::FP32);
  output_data_ptr->setLayout(InferenceEngine::Layout::NCHW);
  //set input and output layer name
  input_ = input_info_map.begin()->first;
  output_ = output_info_map.begin()->first;
}

void ValidatedEmotionsClassificationNetwork::checkLayerProperty(
    const InferenceEngine::CNNNetReader::Ptr &net_reader) {
  slog::info << "Checking Emotions Detection outputs" << slog::endl;
  InferenceEngine::OutputsDataMap output_info(
      net_reader->getNetwork().getOutputsInfo());
  InferenceEngine::DataPtr emotions_output_ptr = output_info.begin()->second;
  //output layer should be SoftMax type
  if (emotions_output_ptr->getCreatorLayer().lock()->type != "SoftMax") {
    throw std::logic_error(
        "In Emotions Recognition network, Emotion layer ("
            + emotions_output_ptr->getCreatorLayer().lock()->name +
            ") should be a SoftMax, but was: " +
            emotions_output_ptr->getCreatorLayer().lock()->type);
  }
  slog::info << "Emotions layer: "
             << emotions_output_ptr->getCreatorLayer().lock()->name
             << slog::endl;
};

const std::string
ValidatedEmotionsClassificationNetwork::getNetworkName() const {
  return "Emotions Detection";
};

//Validated Age Gender Classification Network
ValidatedAgeGenderNetwork::ValidatedAgeGenderNetwork(
    const std::string &model_loc, const std::string &device,
    int input_num, int output_num, int max_batch_size)
    : ValidatedBaseNetwork(model_loc, device,
                           input_num, output_num, max_batch_size){};

void ValidatedAgeGenderNetwork::setLayerProperty(
    InferenceEngine::CNNNetReader::Ptr net_reader) {
  //set input property
  InferenceEngine::InputsDataMap
      input_info_map(net_reader->getNetwork().getInputsInfo());
  InferenceEngine::InputInfo::Ptr input_info = input_info_map.begin()->second;
  input_info->setPrecision(InferenceEngine::Precision::FP32);
  input_info->setLayout(InferenceEngine::Layout::NCHW);
  //set output property
  InferenceEngine::OutputsDataMap
      output_info_map(net_reader->getNetwork().getOutputsInfo());
  auto it = output_info_map.begin();
  InferenceEngine::DataPtr age_output_ptr = (it++)->second;
  InferenceEngine::DataPtr gender_output_ptr = (it++)->second;
  age_output_ptr->setPrecision(InferenceEngine::Precision::FP32);
  age_output_ptr->setLayout(InferenceEngine::Layout::NCHW);
  gender_output_ptr->setPrecision(InferenceEngine::Precision::FP32);
  gender_output_ptr->setLayout(InferenceEngine::Layout::NCHW);
  //set input and output layer name
  input_ = input_info_map.begin()->first;
  output_age_ = age_output_ptr->name;
  output_gender_ = gender_output_ptr->name;
}

void ValidatedAgeGenderNetwork::checkLayerProperty(
    const InferenceEngine::CNNNetReader::Ptr &net_reader) {
  slog::info << "Checking Age Gender Detection outputs" << slog::endl;
  InferenceEngine::OutputsDataMap output_info(
      net_reader->getNetwork().getOutputsInfo());
  auto it = output_info.begin();
  InferenceEngine::DataPtr age_output_ptr = (it++)->second;
  InferenceEngine::DataPtr gender_output_ptr = (it++)->second;
  //output layer of age should be Convolution type
  if (gender_output_ptr->getCreatorLayer().lock()->type == "Convolution") {
    std::swap(age_output_ptr, gender_output_ptr);
  }
  if (age_output_ptr->getCreatorLayer().lock()->type != "Convolution") {
    throw std::logic_error(
        "In Age Gender network, age layer (" +
            age_output_ptr->getCreatorLayer().lock()->name +
            ") should be a Convolution, but was: " +
            age_output_ptr->getCreatorLayer().lock()->type);
  }
  if (gender_output_ptr->getCreatorLayer().lock()->type != "SoftMax") {
    throw std::logic_error(
        "In Age Gender network, gender layer (" +
            gender_output_ptr->getCreatorLayer().lock()->name +
            ") should be a SoftMax, but was: " +
            gender_output_ptr->getCreatorLayer().lock()->type);
  }
  slog::info << "Age layer: " <<
             age_output_ptr->getCreatorLayer().lock()->name <<
             slog::endl;
  slog::info << "Gender layer: " <<
             gender_output_ptr->getCreatorLayer().lock()->name <<
             slog::endl;
};

const std::string ValidatedAgeGenderNetwork::getNetworkName() const {
  return "Age Gender Detection";
};

//Validated Head Pose Network
ValidatedHeadPoseNetwork::ValidatedHeadPoseNetwork(
    const std::string &model_loc, const std::string &device,
    int input_num, int output_num, int max_batch_size)
    : ValidatedBaseNetwork(model_loc, device,
                           input_num, output_num, max_batch_size){};

void ValidatedHeadPoseNetwork::checkLayerProperty(
    const InferenceEngine::CNNNetReader::Ptr &net_reader) {
  slog::info << "Checking Head Pose network outputs" << slog::endl;
  InferenceEngine::OutputsDataMap outputInfo(
      net_reader->getNetwork().getOutputsInfo());
  std::map<std::string, bool> layerNames = {
      {output_angle_r_, false},
      {output_angle_p_, false},
      {output_angle_y_, false}
  };

  for (auto &&output : outputInfo) {
    InferenceEngine::CNNLayerPtr layer =
        output.second->getCreatorLayer().lock();
    if (layerNames.find(layer->name) == layerNames.end()) {
      throw std::logic_error(
          "Head Pose network output layer unknown: " + layer->name
              + ", should be " +
              output_angle_r_ + " or " + output_angle_p_ + " or "
              + output_angle_y_);
    }
    if (layer->type != "FullyConnected") {
      throw std::logic_error("Head Pose network output layer (" + layer->name
                                 + ") has invalid type: " +
          layer->type + ", should be FullyConnected");
    }
    auto fc = dynamic_cast<InferenceEngine::FullyConnectedLayer *>(layer.get());
    if (fc->_out_num != 1) {
      throw std::logic_error("Head Pose network output layer (" + layer->name
                                 + ") has invalid out-size=" +
          std::to_string(fc->_out_num) + ", should be 1");
    }
    layerNames[layer->name] = true;
  }
}

void ValidatedHeadPoseNetwork::setLayerProperty(
    InferenceEngine::CNNNetReader::Ptr net_reader) {
  //set input property
  InferenceEngine::InputsDataMap
      input_info_map(net_reader->getNetwork().getInputsInfo());
  InferenceEngine::InputInfo::Ptr input_info = input_info_map.begin()->second;
  input_info->setPrecision(InferenceEngine::Precision::FP32);
  input_info->setLayout(InferenceEngine::Layout::NCHW);
  //set output property
  InferenceEngine::OutputsDataMap
      output_info_map(net_reader->getNetwork().getOutputsInfo());
  auto it = output_info_map.begin();
  InferenceEngine::DataPtr age_output_ptr = (it++)->second;
  InferenceEngine::DataPtr gender_output_ptr = (it++)->second;
  age_output_ptr->setPrecision(InferenceEngine::Precision::FP32);
  age_output_ptr->setLayout(InferenceEngine::Layout::NCHW);
  gender_output_ptr->setPrecision(InferenceEngine::Precision::FP32);
  gender_output_ptr->setLayout(InferenceEngine::Layout::NCHW);
  //set input and output layer name
  input_ = input_info_map.begin()->first;
}

const std::string ValidatedHeadPoseNetwork::getNetworkName() const {
  return "Head Pose Network";
};