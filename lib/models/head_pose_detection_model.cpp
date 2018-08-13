/**
 * @brief a header file with declaration of HeadPoseDetectionModel class
 * @file head_pose_detection_model.cpp
 */
#include "openvino_service/models/head_pose_detection_model.h"

#include "openvino_service/slog.hpp"

//Validated Head Pose Network
Models::HeadPoseDetectionModel::HeadPoseDetectionModel(
    const std::string &model_loc,
    int input_num, int output_num, int max_batch_size)
    : BaseModel(model_loc, input_num, output_num, max_batch_size){};

void Models::HeadPoseDetectionModel::checkLayerProperty(
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

void Models::HeadPoseDetectionModel::setLayerProperty(
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

const std::string Models::HeadPoseDetectionModel::getModelName() const {
  return "Head Pose Network";
};