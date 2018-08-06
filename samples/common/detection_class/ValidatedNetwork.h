//
// Created by chris on 18-8-2.
//

#ifndef SAMPLES_VALIDATEDCNNNETWORK_H
#define SAMPLES_VALIDATEDCNNNETWORK_H

#include "samples/slog.hpp"
#include "samples/common.hpp"
#include "inference_engine.hpp"

class ValidatedBaseNetwork {
 public:
  ValidatedBaseNetwork(
      const std::string&, const std::string&, size_t, int, int);
  inline std::vector<std::string> &getLabels() { return labels_; }
  inline const std::string getInputLayerName() const {return input_;}
  inline const std::string getOutputLayerName() const {return output_;}

 protected:
  virtual const std::string getNetworkName() const = 0;
  virtual void checkLayerProperty(
      const InferenceEngine::CNNNetReader::Ptr&) = 0;
  virtual void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) = 0;

 private:
  friend class NetworkEngine;
  void networkInit(InferenceEngine::CNNNetReader::Ptr);
  void checkNetworkSize(int, int, InferenceEngine::CNNNetReader::Ptr);
  InferenceEngine::CNNNetReader::Ptr net_reader_;
  std::vector<std::string> labels_;
  size_t max_batch_size_;
  std::string model_loc_;
  std::string device_;
  std::string input_;
  std::string output_;
};

class ValidatedFaceDetectionNetwork : public ValidatedBaseNetwork {
 public:
  ValidatedFaceDetectionNetwork(
      const std::string &, const std::string &, size_t, int, int);

 protected:
  const std::string getNetworkName() const override ;
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override ;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override ;

 private:
  int max_proposal_count_;
  int object_size_;
};

#endif //SAMPLES_VALIDATEDCNNNETWORK_H
