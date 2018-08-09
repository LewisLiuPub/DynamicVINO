// Created by chris on 18-7-12.
//

#include "detection.h"

using namespace InferenceEngine;//TODO need to be deleted to not pollute namespace
using namespace openvino_service;

// utils
template<typename T>
void
matU8ToBlob(const cv::Mat &orig_image, Blob::Ptr &blob, float scale_factor = 1.0,
            int batch_index = 0) {
  SizeVector blob_size = blob->getTensorDesc().getDims();
  const size_t width = blob_size[3];
  const size_t height = blob_size[2];
  const size_t channels = blob_size[1];
  T *blob_data = blob->buffer().as<T *>();

  cv::Mat resized_image(orig_image);
  if (width != orig_image.size().width ||
      height != orig_image.size().height) {
    cv::resize(orig_image, resized_image, cv::Size(width, height));
  }
  int batchOffset = batch_index * width * height * channels;

  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        blob_data[batchOffset + c * width * height + h * width + w] =
            resized_image.at<cv::Vec3b>(h, w)[c] * scale_factor;
      }
    }
  }
}

//BaseInference
BaseInference::BaseInference() = default;

BaseInference::~BaseInference() = default;

void BaseInference::loadEngine(const std::shared_ptr<NetworkEngine> engine) {
  engine_ = engine;
};

template <typename T>
bool BaseInference::enqueue(const cv::Mat &frame, const cv::Rect &,
                            float scale_factor, int batch_index,
                            const std::string &input_name) {
  if (enqueued_frames == max_batch_size_) {
    slog::warn << "Number of " << getName() <<
               "input more than maximum("
               << max_batch_size_
               << ") processed by inference" << slog::endl;
    return false;
  }
  Blob::Ptr input_blob
      = engine_->getRequest()->GetBlob(input_name);
  matU8ToBlob<T>(frame, input_blob, scale_factor, batch_index);
  enqueued_frames += 1;
  return true;
}

bool BaseInference::submitRequest() {
  if (engine_->getRequest() == nullptr) return false;
  if (!enqueued_frames) return false;
  enqueued_frames = 0;
  results_fetched_ = false;
  engine_->getRequest()->StartAsync();
  return true;
}

bool BaseInference::fetchResults() {
  if (results_fetched_) return false;
  results_fetched_ = true;
  return true;
}

// FaceDetection
FaceDetection::FaceDetection(int max_proposal_count, int object_size, double show_output_thresh)
    : max_proposal_count_(max_proposal_count),
      object_size_(object_size),
      show_output_thresh_(show_output_thresh), BaseInference() {
};

FaceDetection::~FaceDetection() = default;

void FaceDetection::loadNetwork(
    const std::shared_ptr<ValidatedFaceDetectionNetwork> network) {
  valid_network_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool
FaceDetection::enqueue(const cv::Mat &frame, const cv::Rect &input_frame_loc) {
  if (width_ == 0 && height_ == 0) {
    width_ = frame.cols;
    height_ = frame.rows;
  }
  if (!BaseInference::enqueue<u_int8_t>(frame, input_frame_loc, 1, 0,
                              valid_network_->getInputName())) {
    return false;
  };
  Result r;
  r.location = input_frame_loc;
  results_.clear();
  results_.emplace_back(r);
  return true;
};

bool FaceDetection::submitRequest() {
  return BaseInference::submitRequest();
};

bool FaceDetection::fetchResults() {
  bool can_fetch = BaseInference::fetchResults();
  if (!can_fetch) return false;
  bool found_result = false;
  results_.clear();
  InferenceEngine::InferRequest::Ptr request = getEngine()->getRequest();
  std::string output = valid_network_->getOutputName();
  const float *detections = request->GetBlob(output)->buffer().as<float *>();
  for (int i = 0; i < max_proposal_count_; i++) {
    float image_id = detections[i * object_size_ + 0];
    Result r;
    auto label_num = static_cast<int>(detections[i * object_size_ + 1]);
    std::vector<std::string> &labels = valid_network_->getLabels();
    r.label = label_num < labels.size() ? labels[label_num] :
              std::string("label #") + std::to_string(label_num);
    r.confidence = detections[i * object_size_ + 2];
    if (r.confidence <= show_output_thresh_) {
      continue;
    }
    found_result = true;
    r.location.x = static_cast<int>(detections[i * object_size_ + 3] *
        width_);
    r.location.y = static_cast<int>(detections[i * object_size_ + 4] *
        height_);
    r.location.width = static_cast<int>(
        detections[i * object_size_ + 5] * width_ - r.location.x);
    r.location.height = static_cast<int>(
        detections[i * object_size_ + 6] * height_ - r.location.y);

    if (image_id < 0) {
      break;
    }
    results_.emplace_back(r);
  }
  if (!found_result) results_.clear();
  return true;
};

void FaceDetection::accepts(std::shared_ptr<BaseOutput> output_visitor) {
  for (auto &result : results_) {
    output_visitor->prepareData(result);
  }
}

const int FaceDetection::getResultsLength() const {
  return (int)results_.size();
};

const InferenceResult::Result FaceDetection::getLocationResult(int idx) const {
  return results_[idx];
};

const std::string FaceDetection::getName() const {
  return valid_network_->getNetworkName();
};

// Emotions Detection
EmotionsDetection::EmotionsDetection()
    : BaseInference() {};

EmotionsDetection::~EmotionsDetection() = default;

void EmotionsDetection::loadNetwork(
    const std::shared_ptr<ValidatedEmotionsClassificationNetwork> network) {
  valid_network_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool EmotionsDetection::enqueue(const cv::Mat &frame,
                                const cv::Rect &input_frame_loc) {
  if (getEnqueuedNum() == 0) { results_.clear(); }
  bool succeed = BaseInference::enqueue<float>(
      frame, input_frame_loc, 1, getResultsLength(),
      valid_network_->getInputName());
  if (!succeed ) return false;
  Result r;
  r.location = input_frame_loc;
  results_.emplace_back(r);
  return true;
}

bool EmotionsDetection::submitRequest() {
  return BaseInference::submitRequest();
}

bool EmotionsDetection::fetchResults() {
  bool can_fetch = BaseInference::fetchResults();
  if (!can_fetch) return false;
  int label_length = static_cast<int>(valid_network_->getLabels().size());
  std::string output_name = valid_network_->getOutputName();
  Blob::Ptr emotions_blob = getEngine()->getRequest()->GetBlob(output_name);
  /** emotions vector must have the same size as number of channels
      in model output. Default output format is NCHW so we check index 1 */
  long num_of_channels = emotions_blob->getTensorDesc().getDims().at(1);
  if (num_of_channels != label_length) {
    throw std::logic_error("Output size (" + std::to_string(num_of_channels) +
        ") of the Emotions Recognition network is not equal "
        "to used emotions vector size (" +
        std::to_string(label_length )+ ")");
  }
  /** we identify an index of the most probable emotion in output array
      for idx image to return appropriate emotion name */
  auto emotions_values = emotions_blob->buffer().as<float *>();
  for (int idx = 0; idx < results_.size(); ++idx) {
    auto output_idx_pos = emotions_values + idx;
    long max_prob_emotion_idx =
        std::max_element(output_idx_pos, output_idx_pos + label_length) -
            output_idx_pos;
    results_[idx].label = valid_network_->getLabels()[max_prob_emotion_idx];
  }
  return true;
};

void EmotionsDetection::accepts(std::shared_ptr<BaseOutput> output_visitor) {
  for (auto &result : results_) {
    output_visitor->prepareData(result);
  }
};

const int EmotionsDetection::getResultsLength() const {
  return (int)results_.size();
};

const InferenceResult::Result
EmotionsDetection::getLocationResult(int idx) const {
  return results_[idx];
};

const std::string EmotionsDetection::getName() const {
  return valid_network_->getNetworkName();
};

// AgeGender Detection
AgeGenderDetection::AgeGenderDetection()
    : BaseInference() {};

AgeGenderDetection::~AgeGenderDetection() = default;

void AgeGenderDetection::loadNetwork(
    std::shared_ptr<ValidatedAgeGenderNetwork> network) {
  valid_network_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool AgeGenderDetection::enqueue(const cv::Mat &frame,
                                 const cv::Rect &input_frame_loc) {
  if (getEnqueuedNum() == 0) { results_.clear(); }
  bool succeed = BaseInference::enqueue<float>(
      frame, input_frame_loc, 1, getResultsLength(),
      valid_network_->getInputName());
  if (!succeed ) return false;
  Result r;
  r.location = input_frame_loc;
  results_.emplace_back(r);
  return true;
}

bool AgeGenderDetection::submitRequest() {
  return BaseInference::submitRequest();
}

bool AgeGenderDetection::fetchResults() {
  bool can_fetch = BaseInference::fetchResults();
  if (!can_fetch) return false;
  auto request = getEngine()->getRequest();
  Blob::Ptr genderBlob = request->GetBlob(valid_network_->getOutputGenderName());
  Blob::Ptr ageBlob = request->GetBlob(valid_network_->getOutputAgeName());

  for (int i = 0; i < results_.size(); ++i) {
    results_[i].age = ageBlob->buffer().as<float *>()[i] * 100;
    results_[i].male_prob = genderBlob->buffer().as<float *>()[i * 2 + 1];
  }
  return true;
};

void AgeGenderDetection::accepts(std::shared_ptr<BaseOutput> output_visitor) {
  for (auto &result : results_) {
    output_visitor->prepareData(result);
  }
};

const int AgeGenderDetection::getResultsLength() const {
  return (int)results_.size();
};

const InferenceResult::Result
AgeGenderDetection::getLocationResult(int idx) const {
  return results_[idx];
};

const std::string AgeGenderDetection::getName() const {
  return valid_network_->getNetworkName();
};

//Head Pose Detection
HeadPoseDetection::HeadPoseDetection()
    : BaseInference() {};

HeadPoseDetection::~HeadPoseDetection() = default;

void HeadPoseDetection::loadNetwork(
    std::shared_ptr<ValidatedHeadPoseNetwork> network) {
  valid_network_ = network;
  setMaxBatchSize(network->getMaxBatchSize());
}

bool HeadPoseDetection::enqueue(const cv::Mat &frame,
                                 const cv::Rect &input_frame_loc) {
  if (getEnqueuedNum() == 0) { results_.clear(); }
  bool succeed = BaseInference::enqueue<float>(
      frame, input_frame_loc, 1, getResultsLength(),
      valid_network_->getInputName());
  if (!succeed ) return false;
  Result r;
  r.location = input_frame_loc;
  results_.emplace_back(r);
  return true;
}

bool HeadPoseDetection::submitRequest() {
  return BaseInference::submitRequest();
}

bool HeadPoseDetection::fetchResults() {
  bool can_fetch = BaseInference::fetchResults();
  if (!can_fetch) return false;
  auto request = getEngine()->getRequest();
  Blob::Ptr angle_r = request->GetBlob(valid_network_->getOutputOutputAngleR());
  Blob::Ptr angle_p = request->GetBlob(valid_network_->getOutputOutputAngleP());
  Blob::Ptr angle_y = request->GetBlob(valid_network_->getOutputOutputAngleY());

  for (int i = 0; i < getResultsLength(); ++i) {
    results_[i].angle_r = angle_r->buffer().as<float *>()[i];
    results_[i].angle_p = angle_p->buffer().as<float *>()[i];
    results_[i].angle_y = angle_y->buffer().as<float *>()[i];
  }
  return true;
};

void HeadPoseDetection::accepts(std::shared_ptr<BaseOutput> output_visitor) {
  for (auto &result : results_) {
    output_visitor->prepareData(result);
  }
};

const int HeadPoseDetection::getResultsLength() const {
  return (int)results_.size();
};

const InferenceResult::Result
HeadPoseDetection::getLocationResult(int idx) const {
  return results_[idx];
};

const std::string HeadPoseDetection::getName() const {
  return valid_network_->getNetworkName();
};
