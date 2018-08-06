//
// Created by chris on 18-8-2.
//

#ifndef SAMPLES_ENGINE_H
#define SAMPLES_ENGINE_H

#include "validated_network.h"
#include "inference_engine.hpp"

class NetworkEngine {
 public:
  NetworkEngine(InferenceEngine::InferencePlugin*,
                const ValidatedBaseNetwork&);
  inline InferenceEngine::InferRequest::Ptr &getRequest() { return request_; }
  template<typename T>
  void setCompletionCallback(const T & callbackToSet) {
    request_->SetCompletionCallback(callbackToSet);
  }

 private:
  InferenceEngine::InferRequest::Ptr request_;

};

#endif //SAMPLES_ENGINE_H
