/**
 * @brief a header file with definition of Engine class
 * @file engine.cpp
 */
#include "openvino_service/engines/engine.h"

Engines::Engine::Engine(
    InferenceEngine::InferencePlugin plg,
    const Models::BaseModel::Ptr base_model) {
  request_ = (plg.LoadNetwork(base_model->net_reader_->getNetwork(), {}))
      .CreateInferRequestPtr();
};