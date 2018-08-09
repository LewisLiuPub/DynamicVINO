//
// Created by chris on 18-8-2.
//

#include "openvino_service/engines/engine.h"

NetworkEngine::NetworkEngine(
    InferenceEngine::InferencePlugin *plg,
    const ValidatedBaseNetwork & validated_network) {
  request_ = (plg->LoadNetwork(validated_network.net_reader_->getNetwork(), {}))
      .CreateInferRequestPtr();
};