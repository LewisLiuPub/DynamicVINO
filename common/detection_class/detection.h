//
// Created by chris on 18-7-12.
//

#ifndef SAMPLES_DETECTION_H
#define SAMPLES_DETECTION_H

#include "validated_network.h"
#include "engine.h"
#include "result.h"
#include "output.h"
#include "opencv2/opencv.hpp"
#include "inference_engine.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"
#include <memory>

/**
 * @brief This namespace contains all classes needed for detection jobs.
 */
namespace openvino_service {
/**
 * @brief base class for detection
 */
class BaseInference {
 public:
  explicit BaseInference();
  virtual ~BaseInference();
  void loadEngine(std::shared_ptr<NetworkEngine>);
  inline const std::shared_ptr<NetworkEngine> getEngine() const {
    return engine_;
  }
  inline const int getEnqueuedNum() const { return enqueued_frames; }
  virtual bool enqueue(const cv::Mat &frame, const cv::Rect &) = 0;
  virtual bool submitRequest();
  /**
   * @brief This function will add the content of frame into input blob of the network.
   * @return bool, whether the Inference object fetches a result this time
   */
  virtual bool fetchResults();
  virtual void accepts(std::shared_ptr<BaseOutput> output_visitor) = 0;
  virtual const int getResultsLength() const = 0;
  virtual const InferenceResult::Result
  getLocationResult(int idx) const = 0;
  virtual const std::string getName() const = 0;

 protected:
  template <typename T>
  bool enqueue(const cv::Mat &frame, const cv::Rect &, float, int,
               const std::string&);
  inline void setMaxBatchSize(int max_batch_size) {
    max_batch_size_ = max_batch_size;
  }

 private:
  std::shared_ptr<NetworkEngine> engine_;
  int max_batch_size_ = 1;
  int enqueued_frames = 0;
  bool results_fetched_ = false;
};

//Face Detection
class FaceDetection : public BaseInference {
 public:
  using Result = InferenceResult::FaceDetectionResult;
  explicit FaceDetection(int, int, double);
  ~FaceDetection() override ;
  void loadNetwork(std::shared_ptr<ValidatedFaceDetectionNetwork>);
  bool enqueue(const cv::Mat &, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<BaseOutput> output_visitor) override ;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override ;
  const std::string getName() const override;

 private:
  std::shared_ptr<ValidatedFaceDetectionNetwork> valid_network_;
  std::vector<Result> results_;
  int width_ = 0;
  int height_ = 0;
  int max_proposal_count_;
  int object_size_;
  double show_output_thresh_ = 0;
};

//Emotions Detection
class EmotionsDetection : public BaseInference {
 public:
  using Result = InferenceResult::EmotionsResult;
  explicit EmotionsDetection();
  ~EmotionsDetection() override ;
  void loadNetwork(std::shared_ptr<ValidatedEmotionsClassificationNetwork>);
  bool enqueue(const cv::Mat &, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<BaseOutput> output_visitor) override ;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override ;
  const std::string getName() const override;

 private:
  std::shared_ptr<ValidatedEmotionsClassificationNetwork> valid_network_;
  std::vector<Result> results_;
};

// AgeGender Detection
class AgeGenderDetection : public BaseInference {
 public:
  using Result = InferenceResult::AgeGenderResult;
  explicit AgeGenderDetection();
  ~AgeGenderDetection() override ;
  void loadNetwork(std::shared_ptr<ValidatedAgeGenderNetwork>);
  bool enqueue(const cv::Mat &frame, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<BaseOutput> output_visitor) override ;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override ;
  const std::string getName() const override ;

 private:
  std::shared_ptr<ValidatedAgeGenderNetwork> valid_network_;
  std::vector<Result> results_;
};

// Head Pose Detection
class HeadPoseDetection : public BaseInference {
 public:
  using Result = InferenceResult::HeadPoseResult;
  explicit HeadPoseDetection();
  ~HeadPoseDetection() override ;
  void loadNetwork(std::shared_ptr<ValidatedHeadPoseNetwork>);
  bool enqueue(const cv::Mat &frame, const cv::Rect &) override;
  bool submitRequest() override;
  bool fetchResults() override;
  void accepts(std::shared_ptr<BaseOutput> output_visitor) override ;
  const int getResultsLength() const override;
  const InferenceResult::Result
  getLocationResult(int idx) const override ;
  const std::string getName() const override ;

 private:
  std::shared_ptr<ValidatedHeadPoseNetwork> valid_network_;
  std::vector<Result> results_;
};
}

#endif //SAMPLES_DETECTION_H