/**
 * @brief A header file with declaration for NetworkEngine class
 * @file engine.h
 */
#pragma once

#include "inference_engine.hpp"
#include "openvino_service/models/base_model.h"

/**
 * @class NetworkEngine
 * @brief This class is used to get the infer request 
 * from a inference plugin and an inference network
 */
namespace Engines {
class Engine {
 public:
  /**
   * @brief Create an NetworkEngine instance 
   * from a inference plugin and an inference network.
   */
  Engine(InferenceEngine::InferencePlugin, Models::BaseModel::Ptr );
  /**
   * @brief Get the inference request this instance holds.
   * @return The inference request this instance holds.
   */
  inline InferenceEngine::InferRequest::Ptr &getRequest() { return request_; }
  /**
   * @brief Set a callback function for the infer request. 
   * @param[in] callbackToSet A lambda function as callback function.
   * The callback function will be called when request is finished.
   */
  template<typename T>
  void setCompletionCallback(const T &callbackToSet) {
    request_->SetCompletionCallback(callbackToSet);
  }

 private:
  InferenceEngine::InferRequest::Ptr request_;
};

}

