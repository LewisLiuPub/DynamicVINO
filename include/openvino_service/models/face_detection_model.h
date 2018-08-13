/**
 * @brief A header file with declaration for FaceDetectionModel Class
 * @file face_detection_model.h
 */

#ifndef OPENVINO_PIPELINE_LIB_FACE_DETECTION_MODEL_H
#define OPENVINO_PIPELINE_LIB_FACE_DETECTION_MODEL_H

#include <openvino_service/models/base_model.h>

namespace Models {

class FaceDetectionModel: public BaseModel {
 public:
  FaceDetectionModel(
      const std::string &, int, int, int);
  inline const int getMaxProposalCount() { return max_proposal_count_; }
  inline const int getObjectSize() { return object_size_; }
  inline const std::string getInputName() { return input_; }
  inline const std::string getOutputName() { return output_; }
  const std::string getModelName() const override;

 protected:
  void checkLayerProperty(const InferenceEngine::CNNNetReader::Ptr &) override;
  void setLayerProperty(InferenceEngine::CNNNetReader::Ptr) override;

 private:
  int max_proposal_count_;
  int object_size_;
  std::string input_;
  std::string output_;
};

}

#endif //OPENVINO_PIPELINE_LIB_FACE_DETECTION_MODEL_H
