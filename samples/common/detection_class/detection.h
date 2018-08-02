//
// Created by chris on 18-7-12.
//

#ifndef SAMPLES_DETECTION_H
#define SAMPLES_DETECTION_H

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>

/**
 * @brief This namespace contains all classes needed for detection jobs.
 */
namespace DetectionClass {
/**
 * @brief base class for detection
 */
class Detection {
 public:
  struct Result {
    std::string label = "";
    float confidence = -1;
    cv::Rect location;
    float angle_y = -1;
    float angle_p = -1;
    float angle_r = -1;
    float age = -1;
    float male_prob = -1;
  };

  Detection(const std::string &model_loc, const std::string &device,
            size_t max_batch_size);

  virtual ~Detection();

  /**
   * @brief Read model into network reader, initialize and config the network' s input and output.
   * @return the configured network object
   */
  InferenceEngine::CNNNetwork read();

  virtual void enqueue(const cv::Mat &, const cv::Rect &) = 0;

  virtual void submitRequest();

  virtual void wait();

  inline size_t getResultsLength() { return results_.size(); };

  bool enabled();

  void load(InferenceEngine::InferencePlugin &plg);

  inline std::vector<std::string> &
  getLabels() { return labels_; } //TODO can be protected?

  void printPerformanceCounts();

  virtual void fetchResults() = 0;

  inline InferenceEngine::InferRequest::Ptr &
  getRequest() { return request_; } //TODO can be protected?

  template<typename T>
  inline void setCompletionCallback(const T &callbackToSet) {
    if (!getRequest()) {
      setRequest(getNetwork().CreateInferRequestPtr());
    }
    request_->SetCompletionCallback(callbackToSet);
  };

  Result
  &operator[](int idx) { return results_[idx]; };

  inline std::vector<Detection::Result> &getAllResults() { return results_; }

  virtual const std::string getName() const = 0;

 protected:
  //setter
  inline void setMaxBatchSize(
      size_t max_batch_size) { max_batch_size_ = max_batch_size; }

  inline void
  setModelLoc(const std::string &model_loc) { model_loc_ = model_loc; }

  inline void
  setLabels(const std::vector<std::string> &labels) { labels_ = labels; }

  inline void setDevice(const std::string &device) { device_ = device; };

  inline void setRequest(
      const InferenceEngine::InferRequest::Ptr &request) {
    request_ = request;
  };

  inline void setNetwork(
      InferenceEngine::ExecutableNetwork &network) { network_ = network; }

  inline void setRawOutput(bool raw_output) { raw_output_ = raw_output; }

  inline void setResult(int idx, const Result &result) {
    results_[idx] = result;
  }

  //getter
  inline const size_t getMaxBatchSize() const { return max_batch_size_; }

  inline const std::string getModelLoc() const { return model_loc_; }

  inline const std::string getDevice() const { return device_; }

  inline InferenceEngine::ExecutableNetwork &
  getNetwork() { return network_; }

  inline bool getRawOutput() { return raw_output_; }

  inline void clearResults() {
    results_.clear();
  }

  inline void addResultWithGivenBoundingBox(const cv::Rect &rect) {
    Result result;
    result.location = rect;
    results_.emplace_back(result);
  };

  inline void addResult(Result &result) {
    results_.emplace_back(result);
  };

  //exclusive for Read()
  void networkInit(InferenceEngine::CNNNetReader *);

  virtual void
  initAndCheckInput(InferenceEngine::CNNNetReader *) = 0;

  virtual void
  initAndCheckOutput(InferenceEngine::CNNNetReader *) = 0;

 private:
  mutable bool enablingChecked_ = false;
  mutable bool enable_ = false;
  std::vector<Result> results_;
  size_t max_batch_size_;
  std::string model_loc_;
  std::string device_;
  std::vector<std::string> labels_;
  InferenceEngine::ExecutableNetwork network_;
  InferenceEngine::InferencePlugin *plugin_;
  InferenceEngine::InferRequest::Ptr request_ = nullptr;
  bool raw_output_ = false;
};

// Face Detection
class FaceDetection : public Detection {
 public:
  using Result = Detection::Result;

  FaceDetection(const std::string &model_loc, const std::string &device,
                double show_output_thresh);

  /**
   * @brief This function will add the content of frame into input blob of the network.
   * @param frame
   */
  void enqueue(const cv::Mat &, const cv::Rect &) override;

  void submitRequest() override;

  void fetchResults() override;

 protected:
  //void networkInit(InferenceEngine::CNNNetReader *net_reader) override;

  void
  initAndCheckInput(InferenceEngine::CNNNetReader *net_reader) override;

  void
  initAndCheckOutput(InferenceEngine::CNNNetReader *net_reader) override;

  const std::string getName() const override { return "Face Detection"; };
 private:
  std::string input_;
  std::string output_;
  int width_ = 0;
  int height_ = 0;
  int enqueued_frames = 0;
  int max_proposal_count_;
  int object_size_;
  bool results_fetched_ = false;
  double show_output_thresh_ = 0;
};

// Emotions Detection
class EmotionsDetection : public Detection {
 public:
  using Result = Detection::Result;

  EmotionsDetection(const std::string &model_loc,
                    const std::string &device, size_t max_batch_size);

  /**
   * @brief This function will add the content of frame into input blob of the network
   * @param frame
   */
  void enqueue(const cv::Mat &frame, const cv::Rect &) override;

  void submitRequest() override;

  void fetchResults() override;

  size_t getLabelsLength();

 protected:
  void
  initAndCheckInput(InferenceEngine::CNNNetReader *net_reader) override;

  void
  initAndCheckOutput(InferenceEngine::CNNNetReader *net_reader) override;

  const std::string
  getName() const override { return "Emotions Detection"; };

 private:
  std::vector<Result> results_;
  int enqueued_faces_num_ = 0;
  std::string input_;
  std::string output_;
  bool results_fetched_;
};

// HeadPose Detection
class HeadPoseDetection : public Detection {
 public:
  using Result = Detection::Result;

  HeadPoseDetection(const std::string &model_loc,
                    const std::string &device, size_t max_batch_size);
  /**
   * @brief This function will add the content of frame into input blob of the network
   * @param frame
   */
  void enqueue(const cv::Mat &frame, const cv::Rect &) override;

  void submitRequest() override;

  void fetchResults() override;

  void buildCameraMatrix(int cx, int cy, float focalLength);

  void drawAxes(cv::Mat &frame,
                cv::Point3f cpoint,
                Result head_pose,
                float scale);

 protected:
  void
  initAndCheckInput(InferenceEngine::CNNNetReader *net_reader) override;

  void
  initAndCheckOutput(InferenceEngine::CNNNetReader *net_reader) override;

  const std::string
  getName() const override { return "HeadPose Detection"; };

 private:
  std::vector<Result> results_;
  int enqueued_faces_num_ = 0;
  std::string input_;
  std::string output_;
  bool results_fetched_;
  std::string outputAngleR = "angle_r_fc";
  std::string outputAngleP = "angle_p_fc";
  std::string outputAngleY = "angle_y_fc";
  cv::Mat camera_matrix_;
};

// AgeGender Detection
class AgeGenderDetection : public Detection {
 public:
  using Result = Detection::Result;

  AgeGenderDetection(const std::string &model_loc,
                     const std::string &device, size_t max_batch_size);

  /**
   * @brief This function will add the content of frame into input blob of the network
   * @param frame
   */
  void enqueue(const cv::Mat &frame, const cv::Rect &) override;

  void submitRequest() override;

  void fetchResults() override;

  //size_t getLabelsLength();

 protected:
  void
  initAndCheckInput(InferenceEngine::CNNNetReader *net_reader) override;

  void
  initAndCheckOutput(InferenceEngine::CNNNetReader *net_reader) override;

  const std::string
  getName() const override { return "AgeGender Detection"; };
 private:
  std::vector<Result> results_;
  int enqueued_faces_num_ = 0;
  std::string input_;
  std::string output_age_;
  std::string output_gender_;
  bool results_fetched_;
};

}
#endif //SAMPLES_DETECTION_H