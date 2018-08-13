/**
 * @brief A header file with declaration for EmotionDetectionModel Class
 * @file emotion_detection_model.h
 */

#ifndef OPENVINO_PIPELINE_LIB_EMOTION_DETECTION_MODEL_H
#define OPENVINO_PIPELINE_LIB_EMOTION_DETECTION_MODEL_H

#include "openvino_service/models/base_model.h"

namespace Models {

class EmotionDetectionModel : public BaseModel {
 public:
  EmotionDetectionModel(
      const std::string &, int, int, int);
  inline const std::string getInputName() { return input_; }
  inline const std::string getOutputName() { return output_; }
  const std::string getModelName() const override;

 protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override;

 private:
  std::string input_;
  std::string output_;
};

}

#endif //OPENVINO_PIPELINE_LIB_EMOTION_DETECTION_MODEL_H
