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
      const std::string&, const std::string&, int, int, int);
  inline std::vector<std::string> &getLabels() { return labels_; }
  inline const std::vector<std::string> getLabels() const { return labels_; }
  inline const int getMaxBatchSize() const { return max_batch_size_;}
  void networkInit();
  virtual const std::string getNetworkName() const = 0;

 protected:
  virtual void checkLayerProperty(
      const InferenceEngine::CNNNetReader::Ptr&) = 0;
  virtual void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) = 0;
 private:
  friend class NetworkEngine;

  void checkNetworkSize(int, int, InferenceEngine::CNNNetReader::Ptr);
  InferenceEngine::CNNNetReader::Ptr net_reader_;
  std::vector<std::string> labels_;
  int input_num_;
  int output_num_;
  int max_batch_size_;
  std::string model_loc_;
  std::string device_;
};

class ValidatedFaceDetectionNetwork : public ValidatedBaseNetwork {
 public:
  ValidatedFaceDetectionNetwork(
      const std::string &, const std::string &, int, int, int);
  inline const int getMaxProposalCount() { return max_proposal_count_; }
  inline const int getObjectSize() { return object_size_; }
  inline const std::string getInputName() {return input_;}
  inline const std::string getOutputName() {return output_;}
  const std::string getNetworkName() const override ;

 protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override ;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override ;

 private:
  int max_proposal_count_;
  int object_size_;
  std::string input_;
  std::string output_;
};

class ValidatedEmotionsClassificationNetwork : public ValidatedBaseNetwork {
 public:
  ValidatedEmotionsClassificationNetwork(
      const std::string &, const std::string &, int, int, int);
  inline const std::string getInputName() {return output_;}
  inline const std::string getOutputName() {return output_;}
  const std::string getNetworkName() const override ;

 protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override ;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override ;

 private:
  std::string input_;
  std::string output_;
};

class ValidatedAgeGenderNetwork : public ValidatedBaseNetwork {
 public:
  ValidatedAgeGenderNetwork(
      const std::string &, const std::string &, int, int, int);
  inline const std::string getInputName() const {return input_;}
  inline const std::string getOutputAgeName() const {return output_age_;}
  inline const std::string getOutputGenderName() const {return output_gender_;}
  const std::string getNetworkName() const override ;

 protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override ;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override ;

 private:
  std::string input_;
  std::string output_age_;
  std::string output_gender_;
};
#endif //SAMPLES_VALIDATEDCNNNETWORK_H
