#include "openvino_service/engines/engine.h"

Engines::Engine::Engine(
    InferenceEngine::InferencePlugin plg,
    const Models::BaseModel::Ptr base_model) {
  request_ = (plg.LoadNetwork(base_model->net_reader_->getNetwork(), {}))
      .CreateInferRequestPtr();
};