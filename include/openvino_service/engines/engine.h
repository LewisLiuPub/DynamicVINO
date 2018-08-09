//
// Created by chris on 18-8-2.
//
/**
 * @brief a header file with definition for NetworkEngine class
 * @file engine.h
 */
#ifndef SAMPLES_ENGINE_H
#define SAMPLES_ENGINE_H

#include "openvino_service/models/validated_network.h"
#include "inference_engine.hpp"
/**
 * @class NetworkEngine
 * @brief This class is used to get the infer request 
 * from a inference plugin and an inference network
 */
class NetworkEngine {
 public:
  /**
   * @brief Create an NetworkEngine instance 
   * from a inference plugin and an inference network.
   */
  NetworkEngine(InferenceEngine::InferencePlugin*,
                const ValidatedBaseNetwork&);
  /**
   * @brief Get the inference request this instance holds.
   */
  inline InferenceEngine::InferRequest::Ptr &getRequest() { return request_; }
  /**
   * @brief Set a callback function for the infer request. 
   * @param[in] callbackToSet A lambda function as callback function.
   * The callback function will be called when request is finished.
   */
  template<typename T>
  void setCompletionCallback(const T & callbackToSet) {
    request_->SetCompletionCallback(callbackToSet);
  }

 private:
  InferenceEngine::InferRequest::Ptr request_;

};

#endif //SAMPLES_ENGINE_H
