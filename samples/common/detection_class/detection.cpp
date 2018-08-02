// Created by chris on 18-7-12.
//

#include "detection.h"

using namespace InferenceEngine;//TODO need to be deleted to not pollute namespace
using namespace InferenceClass;

// utils
template<typename T>
void
matU8ToBlob(const cv::Mat &orig_image, Blob::Ptr &blob, float scaleFactor = 1.0,
            int batchIndex = 0) {
  SizeVector blobSize = blob->getTensorDesc().getDims();
  const size_t width = blobSize[3];
  const size_t height = blobSize[2];
  const size_t channels = blobSize[1];
  T *blob_data = blob->buffer().as<T *>();

  cv::Mat resized_image(orig_image);
  if (width != orig_image.size().width ||
      height != orig_image.size().height) {
    cv::resize(orig_image, resized_image, cv::Size(width, height));
  }

  int batchOffset = batchIndex * width * height * channels;

  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        blob_data[batchOffset + c * width * height + h * width + w] =
            resized_image.at<cv::Vec3b>(h, w)[c] * scaleFactor;
      }
    }
  }
}

//BaseInference
BaseInference::BaseInference(
    const std::string &model_loc,
    const std::string &device,
    size_t max_batch_size)
    : model_loc_(model_loc), device_(device), max_batch_size_(max_batch_size) {}

BaseInference::~BaseInference() = default;

void BaseInference::loadNetwork(
    const std::shared_ptr<ValidatedBaseNetwork> model) {
  model_ = model;
}

void BaseInference::loadEngine(const std::shared_ptr<NetworkEngine> engine) {
  engine_ = engine;
};


CNNNetwork BaseInference::read() {
  try {
    CNNNetReader net_reader;
    networkInit(&net_reader);
    initAndCheckInput(&net_reader);
    initAndCheckOutput(&net_reader);
    return net_reader.getNetwork();
  }
  catch (...) {
    throw std::runtime_error("read CNNNetwork error!");
  }
}

bool BaseInference::enabled() {
  if (!enablingChecked_) {
    enable_ = !model_loc_.empty() && !device_.empty();
    if (!enable_) {
      slog::info << getName() << " DISABLED" << slog::endl;
    }
    enablingChecked_ = true;
  }
  return enable_;
}

void BaseInference::wait() {
  if (!enabled() || !request_) return;
  slog::info << request_->Wait(IInferRequest::WaitMode::RESULT_READY)
             << slog::endl << InferenceEngine::StatusCode::OK;
}

void BaseInference::submitRequest() {
  if (!enabled() || request_ == nullptr) return;
  request_->StartAsync();
}

void BaseInference::networkInit(InferenceEngine::CNNNetReader *net_reader) {
  slog::info << "Loading network files for " << getName() << slog::endl;
  //Read network model
  net_reader->ReadNetwork(model_loc_);
  //Set batch size to given max_batch_size_
  slog::info << "Batch size is set to  " << max_batch_size_ << slog::endl;
  net_reader->getNetwork().setBatchSize(max_batch_size_);
  //Extract model name and load it's weights
  std::string bin_file_name = fileNameNoExt(model_loc_) + ".bin";
  net_reader->ReadWeights(bin_file_name);
  //Read labels (if any)
  std::string label_file_name = fileNameNoExt(model_loc_) + ".labels";
  std::ifstream inputFile(label_file_name);
  std::copy(std::istream_iterator<std::string>(inputFile),
            std::istream_iterator<std::string>(),
            std::back_inserter(labels_));
}

void BaseInference::load(InferenceEngine::InferencePlugin &plg) {
  if (enabled()) {
    network_ = plg.LoadNetwork(read(), {});
    plugin_ = &plg;
  }
}

void BaseInference::printPerformanceCounts() {
  if (!enabled()) {
    return;
  }
  slog::info << "Performance counts for " << getName() << slog::endl
             << slog::endl;
  ::printPerformanceCounts(getRequest()->GetPerformanceCounts(), std::cout,
                           false);
}

// FaceDetection
FaceDetection::FaceDetection(const std::string &model_loc,
                             const std::string &device,
                             double show_output_thresh)
    : BaseInference(model_loc, device, 1),
      show_output_thresh_(show_output_thresh) {};

void
FaceDetection::enqueue(const cv::Mat &frame, const cv::Rect &input_frame_loc) {
  if (!enabled()) { return; }
  if (!getRequest()) {
    setRequest(getNetwork().CreateInferRequestPtr());
  }
  width_ = frame.cols;
  height_ = frame.rows;
  Blob::Ptr input_blob = getRequest()->GetBlob(input_);
  matU8ToBlob<uint8_t>(frame, input_blob);
  clearResults();
  enqueued_frames = 1;
}

void FaceDetection::submitRequest() {
  if (!enqueued_frames) return;
  enqueued_frames = 0;
  results_fetched_ = false;
  clearResults();
  BaseInference::submitRequest();
};

void FaceDetection::initAndCheckInput(CNNNetReader *net_reader) {
  slog::info << "Checking Face Detection inputs" << slog::endl;
  InputsDataMap input_info(net_reader->getNetwork().getInputsInfo());
  if (input_info.size() != 1) {
    throw std::logic_error(
        "Face Detection network should have only one input");
  }
  InputInfo::Ptr input_info_first = input_info.begin()->second;
  input_info_first->setPrecision(Precision::U8);
  input_info_first->setLayout(Layout::NCHW);
  input_ = input_info.begin()->first;
}

void FaceDetection::initAndCheckOutput(CNNNetReader *net_reader) {
  slog::info << "Checking Face Detection outputs" << slog::endl;
  OutputsDataMap output_info(net_reader->getNetwork().getOutputsInfo());
  if (output_info.size() != 1) {
    throw std::logic_error(
        "Face Detection network should have only one output");
  }
  DataPtr &output_data_ptr = output_info.begin()->second;
  output_ = output_info.begin()->first;
  const CNNLayerPtr output_layer = net_reader->getNetwork().getLayerByName(
      output_.c_str());
  if (output_layer->type != "DetectionOutput") {
    throw std::logic_error(
        "Face Detection network output layer(" + output_layer->name +
            ") should be DetectionOutput, but was " + output_layer->type);
  }
  if (output_layer->params.find("num_classes") ==
      output_layer->params.end()) {
    throw std::logic_error("Face Detection network output layer (" +
        output_ +
        ") should have num_classes integer attribute");
  }
  const int num_classes = output_layer->GetParamAsInt("num_classes");
  if (getLabels().size() != num_classes) {
    if (getLabels().size() == (num_classes -
        1))  // if network assumes default "background" class, having no label
      getLabels().insert(getLabels().begin(), "fake");
    else
      getLabels().clear();
  }
  const SizeVector output_dims = output_data_ptr->getTensorDesc().getDims();
  max_proposal_count_ = output_dims[2];
  object_size_ = output_dims[3];
  if (object_size_ != 7) {
    throw std::logic_error(
        "Face Detection network output layer should have 7 as a last dimension");
  }
  if (output_dims.size() != 4) {
    throw std::logic_error(
        "Face Detection network output dimensions not compatible shoulld be 4, but was "
            +
                std::to_string(output_dims.size()));
  }
  output_data_ptr->setPrecision(Precision::FP32);
  output_data_ptr->setLayout(Layout::NCHW);
  slog::info << "Loading Face Detection model to the " << getDevice()
             << " plugin" << slog::endl;
}

void FaceDetection::fetchResults() {
  if (!enabled()) return;
  if (results_fetched_) return;
  results_fetched_ = true;
  const float *detections = getRequest()->GetBlob(
      output_)->buffer().as<float *>();

  for (int i = 0; i < max_proposal_count_; i++) {
    float image_id = detections[i * object_size_ + 0];
    Result r;
    int label_num = static_cast<int>(detections[i * object_size_ + 1]);
    r.label = label_num < getLabels().size() ? getLabels()[label_num] :
              std::string("label #") + std::to_string(label_num);
    r.confidence = detections[i * object_size_ + 2];
    if (r.confidence <= show_output_thresh_) {
      continue;
    }
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
    if (getRawOutput()) {
      std::cout << "[" << i << "," << r.label << "] element, prob = "
                << r.confidence <<
                "    (" << r.location.x << "," << r.location.y << ")-("
                << r.location.width << ","
                << r.location.height << ")"
                << ((r.confidence > show_output_thresh_)
                    ? " WILL BE RENDERED!" : "") << std::endl;
    }

    addResult(r);
  }
}

// Emotions Detection
EmotionsDetection::EmotionsDetection(
    const std::string &model_loc,
    const std::string &device,
    size_t max_batch_size) : BaseInference(model_loc, device, max_batch_size) {};

void EmotionsDetection::submitRequest() {
  if (!enqueued_faces_num_) return;
  BaseInference::submitRequest();
  results_fetched_ = false;
  enqueued_faces_num_ = 0;
}

void EmotionsDetection::enqueue(const cv::Mat &frame,
                                const cv::Rect &input_frame_loc_) {
  if (!enabled()) {
    return;
  }
  if (enqueued_faces_num_ == getMaxBatchSize()) {
    slog::warn << "Number of detected faces more than maximum("
               << getMaxBatchSize()
               << ") processed by Emotions detector" << slog::endl;
    return;
  }
  if (!getRequest()) {
    setRequest(getNetwork().CreateInferRequestPtr());
  }
  if (enqueued_faces_num_ == 0) { clearResults(); }
  Blob::Ptr inputBlob = getRequest()->GetBlob(input_);
  matU8ToBlob<float>(frame, inputBlob, 1.0f, enqueued_faces_num_);
  addResultWithGivenBoundingBox(input_frame_loc_);
  ++enqueued_faces_num_;
}

void EmotionsDetection::initAndCheckInput(
    InferenceEngine::CNNNetReader *net_reader) {
  slog::info << "Checking Emotions Recognition inputs" << slog::endl;
  InferenceEngine::InputsDataMap inputInfo(
      net_reader->getNetwork().getInputsInfo());
  if (inputInfo.size() != 1) {
    throw std::logic_error(
        "Emotions Recognition topology should have only one input");
  }
  auto &inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::FP32);
  inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
  input_ = inputInfo.begin()->first;
}

void EmotionsDetection::initAndCheckOutput(
    InferenceEngine::CNNNetReader *net_reader) {
  slog::info << "Checking Emotions Recognition outputs" << slog::endl;
  InferenceEngine::OutputsDataMap output_info(
      net_reader->getNetwork().getOutputsInfo());
  if (output_info.size() != 1) {
    throw std::logic_error(
        "Emotions Recognition network should have one output layer");
  }

  DataPtr emotions_output_ptr = output_info.begin()->second;

  if (emotions_output_ptr->getCreatorLayer().lock()->type != "SoftMax") {
    throw std::logic_error(
        "In Emotions Recognition network, Emotion layer ("
            + emotions_output_ptr->getCreatorLayer().lock()->name +
            ") should be a SoftMax, but was: " +
            emotions_output_ptr->getCreatorLayer().lock()->type);
  }
  slog::info << "Emotions layer: "
             << emotions_output_ptr->getCreatorLayer().lock()->name << slog::endl;
  output_ = emotions_output_ptr->name;
  slog::info << "Loading Emotions Recognition model to the " << getDevice()
             << " plugin" << slog::endl;
}

size_t EmotionsDetection::getLabelsLength() {
  return getLabels().size();
}

void EmotionsDetection::fetchResults() {
  /* vector of supported emotions */
  //static const std::vector<std::string> getLabels() = {"neutral", "happy", "sad", "surprise", "anger"};
  if (!enabled()) return;
  if (results_fetched_) return;
  clearResults();
  results_fetched_ = true;
  auto emotions_vec_size = getLabelsLength();

  Blob::Ptr emotions_blob = getRequest()->GetBlob(output_);
  /* emotions vector must have the same size as number of channels
   * in model output. Default output format is NCHW so we check index 1. */
  long num_of_channels = emotions_blob->getTensorDesc().getDims().at(1);
  if (num_of_channels != getLabels().size()) {
    throw std::logic_error("Output size (" + std::to_string(num_of_channels) +
        ") of the Emotions Recognition network is not equal "
        "to used emotions vector size (" +
        std::to_string(getLabels().size()) + ")");
  }
  /* we identify an index of the most probable emotion in output array
     for idx image to return appropriate emotion name */
  auto emotions_values = emotions_blob->buffer().as<float *>();
  for (int idx = 0; idx < getResultsLength(); ++idx) {
    auto output_idx_pos = emotions_values + idx;
    long max_prob_emotion_idx =
        std::max_element(output_idx_pos, output_idx_pos + emotions_vec_size) -
            output_idx_pos;
    (*this)[idx].label = getLabels()[max_prob_emotion_idx];
  }
};

// HeadPoseDetection
HeadPoseDetection::HeadPoseDetection(
    const std::string &model_loc,
    const std::string &device,
    size_t max_batch_size) : BaseInference(model_loc, device, max_batch_size) {};

void HeadPoseDetection::submitRequest() {
  if (!enqueued_faces_num_) return;
  BaseInference::submitRequest();
  results_fetched_ = false;
  enqueued_faces_num_ = 0;
}

void HeadPoseDetection::enqueue(const cv::Mat &frame,
                                const cv::Rect &input_frame_loc_) {
  slog::info << "HeadPose detection end enqueue" << slog::endl;
  slog::info << "HeadPose detection result size:" << getResultsLength()
             << slog::endl;
  if (!enabled()) {
    return;
  }
  if (enqueued_faces_num_ == getMaxBatchSize()) {
    slog::warn << "Number of detected faces more than maximum("
               << getMaxBatchSize()
               << ") processed by HeadPose detector" << slog::endl;
    return;
  }
  if (!getRequest()) {
    setRequest(getNetwork().CreateInferRequestPtr());
  }
  if (enqueued_faces_num_ == 0) { clearResults(); }
  Blob::Ptr inputBlob = getRequest()->GetBlob(input_);
  matU8ToBlob<float>(frame, inputBlob, 1.0f, enqueued_faces_num_);
  addResultWithGivenBoundingBox(input_frame_loc_);
  ++enqueued_faces_num_;
  slog::info << "HeadPoseDetection end enqueue" << slog::endl;
}

void HeadPoseDetection::initAndCheckInput(
    InferenceEngine::CNNNetReader *net_reader) {
  /** Age Gender network should have one input two outputs **/
  // ---------------------------Check inputs ------------------------------------------------------
  slog::info << "Checking Head Pose Network inputs" << slog::endl;
  InputsDataMap inputInfo(net_reader->getNetwork().getInputsInfo());
  if (inputInfo.size() != 1) {
    throw std::logic_error("Head Pose topology should have only one input");
  }
  InputInfo::Ptr &inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::FP32);
  inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
  input_ = inputInfo.begin()->first;
}

void HeadPoseDetection::initAndCheckOutput(
    InferenceEngine::CNNNetReader *net_reader) {
  slog::info << "Checking Head Pose network outputs" << slog::endl;
  InferenceEngine::OutputsDataMap outputInfo(
      net_reader->getNetwork().getOutputsInfo());
  if (outputInfo.size() != 3) {
    throw std::logic_error("Head Pose network should have 3 outputs");
  }
  std::map<std::string, bool> layerNames = {
      {outputAngleR, false},
      {outputAngleP, false},
      {outputAngleY, false}
  };

  for (auto &&output : outputInfo) {
    CNNLayerPtr layer = output.second->getCreatorLayer().lock();
    if (layerNames.find(layer->name) == layerNames.end()) {
      throw std::logic_error(
          "Head Pose network output layer unknown: " + layer->name
              + ", should be " +
              outputAngleR + " or " + outputAngleP + " or " + outputAngleY);
    }
    if (layer->type != "FullyConnected") {
      throw std::logic_error("Head Pose network output layer (" + layer->name
                                 + ") has invalid type: " +
          layer->type + ", should be FullyConnected");
    }
    auto fc = dynamic_cast<FullyConnectedLayer *>(layer.get());
    if (fc->_out_num != 1) {
      throw std::logic_error("Head Pose network output layer (" + layer->name
                                 + ") has invalid out-size=" +
          std::to_string(fc->_out_num) + ", should be 1");
    }
    layerNames[layer->name] = true;
  }

  slog::info << "Loading Head Pose model to the " << getDevice() << " plugin"
             << slog::endl;

  enabled();
}

void InferenceClass::HeadPoseDetection::buildCameraMatrix(
    int cx, int cy, float focalLength) {
  if (!camera_matrix_.empty()) return;
  camera_matrix_ = cv::Mat::zeros(3, 3, CV_32F);
  camera_matrix_.at<float>(0) = focalLength;
  camera_matrix_.at<float>(2) = static_cast<float>(cx);
  camera_matrix_.at<float>(4) = focalLength;
  camera_matrix_.at<float>(5) = static_cast<float>(cy);
  camera_matrix_.at<float>(8) = 1;
}

void InferenceClass::HeadPoseDetection::fetchResults() {
  Blob::Ptr angleR = getRequest()->GetBlob(outputAngleR);
  Blob::Ptr angleP = getRequest()->GetBlob(outputAngleP);
  Blob::Ptr angleY = getRequest()->GetBlob(outputAngleY);

  for (int i = 0; i < getResultsLength(); ++i) {
    results_[i].angle_r = angleR->buffer().as<float *>()[i];
    results_[i].angle_p = angleP->buffer().as<float *>()[i];
    results_[i].angle_y = angleY->buffer().as<float *>()[i];
  }
}

void InferenceClass::HeadPoseDetection::drawAxes(
    cv::Mat &frame,
    cv::Point3f cpoint,
    InferenceClass::HeadPoseDetection::Result head_pose,
    float scale) {
  double yaw = head_pose.angle_y;
  double pitch = head_pose.angle_p;
  double roll = head_pose.angle_r;

  if (getRawOutput()) {
    std::cout << "Head pose results: yaw, pitch, roll = " << yaw << ";" << pitch
              << ";" << roll << std::endl;
  }

  pitch *= CV_PI / 180.0;
  yaw *= CV_PI / 180.0;
  roll *= CV_PI / 180.0;

  cv::Matx33f Rx(1, 0, 0,
                 0, cos(pitch), -sin(pitch),
                 0, sin(pitch), cos(pitch));
  cv::Matx33f Ry(cos(yaw), 0, -sin(yaw),
                 0, 1, 0,
                 sin(yaw), 0, cos(yaw));
  cv::Matx33f Rz(cos(roll), -sin(roll), 0,
                 sin(roll), cos(roll), 0,
                 0, 0, 1);

  auto r = cv::Mat(Rz * Ry * Rx);
  buildCameraMatrix(frame.cols / 2, frame.rows / 2, 950.0);

  cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F),
      zAxis1(3, 1, CV_32F);

  xAxis.at<float>(0) = 1 * scale;
  xAxis.at<float>(1) = 0;
  xAxis.at<float>(2) = 0;

  yAxis.at<float>(0) = 0;
  yAxis.at<float>(1) = -1 * scale;
  yAxis.at<float>(2) = 0;

  zAxis.at<float>(0) = 0;
  zAxis.at<float>(1) = 0;
  zAxis.at<float>(2) = -1 * scale;

  zAxis1.at<float>(0) = 0;
  zAxis1.at<float>(1) = 0;
  zAxis1.at<float>(2) = 1 * scale;

  cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
  o.at<float>(2) = camera_matrix_.at<float>(0);

  xAxis = r * xAxis + o;
  yAxis = r * yAxis + o;
  zAxis = r * zAxis + o;
  zAxis1 = r * zAxis1 + o;

  cv::Point p1, p2;

  p2.x = static_cast<int>(
      (xAxis.at<float>(0) / xAxis.at<float>(2) * camera_matrix_.at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (xAxis.at<float>(1) / xAxis.at<float>(2) * camera_matrix_.at<float>(4))
          + cpoint.y);
  cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 0, 255), 2);

  p2.x = static_cast<int>(
      (yAxis.at<float>(0) / yAxis.at<float>(2) * camera_matrix_.at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (yAxis.at<float>(1) / yAxis.at<float>(2) * camera_matrix_.at<float>(4))
          + cpoint.y);
  cv::line(frame, cv::Point(cpoint.x, cpoint.y), p2, cv::Scalar(0, 255, 0), 2);

  p1.x = static_cast<int>(
      (zAxis1.at<float>(0) / zAxis1.at<float>(2) * camera_matrix_.at<float>(0))
          + cpoint.x);
  p1.y = static_cast<int>(
      (zAxis1.at<float>(1) / zAxis1.at<float>(2) * camera_matrix_.at<float>(4))
          + cpoint.y);

  p2.x = static_cast<int>(
      (zAxis.at<float>(0) / zAxis.at<float>(2) * camera_matrix_.at<float>(0))
          + cpoint.x);
  p2.y = static_cast<int>(
      (zAxis.at<float>(1) / zAxis.at<float>(2) * camera_matrix_.at<float>(4))
          + cpoint.y);
  cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 2);
  cv::circle(frame, p2, 3, cv::Scalar(255, 0, 0), 2);
}

// AgeGender Detection
AgeGenderDetection::AgeGenderDetection(
    const std::string &model_loc,
    const std::string &device,
    size_t max_batch_size) : BaseInference(model_loc, device, max_batch_size) {};

void AgeGenderDetection::submitRequest() {
  if (!enqueued_faces_num_) return;
  BaseInference::submitRequest();
  results_fetched_ = false;
  enqueued_faces_num_ = 0;
}

void AgeGenderDetection::enqueue(const cv::Mat &frame,
                                 const cv::Rect &input_frame_loc_) {
  if (!enabled()) {
    return;
  }
  if (enqueued_faces_num_ == getMaxBatchSize()) {
    slog::warn << "Number of detected faces more than maximum("
               << getMaxBatchSize()
               << ") processed by AgeGender detector" << slog::endl;
    return;
  }
  if (!getRequest()) {
    setRequest(getNetwork().CreateInferRequestPtr());
  }
  if (enqueued_faces_num_ == 0) { clearResults(); }
  Blob::Ptr inputBlob = getRequest()->GetBlob(input_);
  matU8ToBlob<float>(frame, inputBlob, 1.0f, enqueued_faces_num_);
  addResultWithGivenBoundingBox(input_frame_loc_);
  ++enqueued_faces_num_;
}

void AgeGenderDetection::initAndCheckInput(
    InferenceEngine::CNNNetReader *net_reader) {
  slog::info << "Loading network files for AgeGender" << slog::endl;
  CNNNetReader netReader;
  netReader.ReadNetwork(getModelLoc());
  netReader.getNetwork().setBatchSize(getMaxBatchSize());
  slog::info << "Batch size is set to " << netReader.getNetwork().getBatchSize()
             << " for Age Gender"
             << slog::endl;
  std::string binFileName = fileNameNoExt(getModelLoc()) + ".bin";
  netReader.ReadWeights(binFileName);

  slog::info << "Checking Age Gender inputs" << slog::endl;
  InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
  if (inputInfo.size() != 1) {
    throw std::logic_error("Age gender topology should have only one input");
  }
  InputInfo::Ptr &inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::FP32);
  inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
  input_ = inputInfo.begin()->first;
}

void AgeGenderDetection::initAndCheckOutput(
    InferenceEngine::CNNNetReader *net_reader) {
  slog::info << "Checking Age Gender outputs" << slog::endl;
  OutputsDataMap output_info(
      net_reader->getNetwork().getOutputsInfo());
  if (output_info.size() != 2) {
    throw std::logic_error(
        "Age Gender network should have two output layers");
  }
  auto it = output_info.begin();
  DataPtr age_output_ptr = (it++)->second;
  DataPtr gender_output_ptr = (it++)->second;
  if (gender_output_ptr->getCreatorLayer().lock()->type == "Convolution") {
    std::swap(age_output_ptr, gender_output_ptr);
  }

  if (age_output_ptr->getCreatorLayer().lock()->type != "Convolution") {
    throw std::logic_error("In Age Gender network, age layer ("
                               + age_output_ptr->getCreatorLayer().lock()->name +
        ") should be a Convolution, but was: "
                               + age_output_ptr->getCreatorLayer().lock()->type);
  }
  if (gender_output_ptr->getCreatorLayer().lock()->type != "SoftMax") {
    throw std::logic_error(
        "In Age Gender network, gender layer ("
            + gender_output_ptr->getCreatorLayer().lock()->name +
            ") should be a SoftMax, but was: "
            + gender_output_ptr->getCreatorLayer().lock()->type);
  }
  slog::info << "Age layer: " << age_output_ptr->getCreatorLayer().lock()->name
             << slog::endl;
  slog::info << "Gender layer: " << gender_output_ptr->getCreatorLayer().lock()->name
             << slog::endl;
  output_age_= age_output_ptr->name;
  output_gender_ = gender_output_ptr->name;
  slog::info << "Loading Age Gender model to the " << getDevice() << " plugin"
             << slog::endl;
  enabled();
}

void AgeGenderDetection::fetchResults() {
  if (!enabled()) return;
  if (results_fetched_) return;
  results_fetched_ = true;

  Blob::Ptr genderBlob = getRequest()->GetBlob(output_gender_);
  Blob::Ptr ageBlob = getRequest()->GetBlob(output_age_);

  for (int i = 0; i < getResultsLength(); ++i) {
    (*this)[i].age = ageBlob->buffer().as<float *>()[i] * 100;
    (*this)[i].male_prob = genderBlob->buffer().as<float *>()[i * 2 + 1];
  }
}