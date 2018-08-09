//
// Created by chris on 18-8-9.
//

#include "openvino_service/inferences/base_inference.h"

//BaseInference
openvino_service::BaseInference::BaseInference() = default;

openvino_service::BaseInference::~BaseInference() = default;

void openvino_service::BaseInference::loadEngine(
    const std::shared_ptr<NetworkEngine> engine) {
  engine_ = engine;
};

bool openvino_service::BaseInference::submitRequest() {
  if (engine_->getRequest() == nullptr) return false;
  if (!enqueued_frames) return false;
  enqueued_frames = 0;
  results_fetched_ = false;
  engine_->getRequest()->StartAsync();
  return true;
}

bool openvino_service::BaseInference::fetchResults() {
  if (results_fetched_) return false;
  results_fetched_ = true;
  return true;
}