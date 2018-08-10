//
// Created by chris on 18-8-9.
//

#ifndef OPENVINO_PIPELINE_LIB_AGE_GENDER_DETECTION_MODEL_H
#define OPENVINO_PIPELINE_LIB_AGE_GENDER_DETECTION_MODEL_H

#include "openvino_service/models/base_model.h"

namespace Models {

class AgeGenderDetectionModel : public BaseModel {
 public:
  AgeGenderDetectionModel(
      const std::string &, int, int, int);
  inline const std::string getInputName() const {return input_;}
  inline const std::string getOutputAgeName() const {return output_age_;}
  inline const std::string getOutputGenderName() const {return output_gender_;}
  const std::string getModelName() const override ;

 protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override ;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override ;

 private:
  std::string input_;
  std::string output_age_;
  std::string output_gender_;
};

}

#endif //OPENVINO_PIPELINE_LIB_AGE_GENDER_DETECTION_MODEL_H
