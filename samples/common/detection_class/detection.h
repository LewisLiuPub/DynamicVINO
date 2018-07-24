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
        struct Result {};

        Detection(const std::string &model_loc, const std::string &device, int max_batch_size);
        virtual ~Detection();
        /**
         * @brief Read model into network reader, initialize and config the network' s input and output.
         * @return the configured network object
         */
        InferenceEngine::CNNNetwork read();

        virtual void enqueue(const cv::Mat &frame) = 0;

        virtual void submitRequest();

        virtual void wait();

        virtual size_t getResultsLength() = 0;

        bool enabled() const;

        void load(InferenceEngine::InferencePlugin &plg);

        inline std::vector<std::string> &getLabels() { return labels_; } //TODO can be protected?

        void printPerformanceCounts();

        virtual void fetchResults() = 0;

        inline InferenceEngine::InferRequest::Ptr &getRequest() { return request_; } //TODO can be protected?

        template<typename T>
        void setCompletionCallback(const T & callbackToSet);

        virtual inline const Result &getResults(int) const = 0;

        virtual const Result* getResultPtr(int idx) const = 0;

    protected:
        //setter
        inline void setMaxBatchSize(int max_batch_size) { max_batch_size_ = max_batch_size; }

        inline void setModelLoc(const std::string &model_loc) { model_loc_ = model_loc; }

        inline void setLabels(const std::vector<std::string> &labels) { labels_ = labels; }

        inline void setDevice(const std::string &device) { device_ = device; };

        inline void setRequest(const InferenceEngine::InferRequest::Ptr &request) { request_ = request; };

        inline void setNetwork(InferenceEngine::ExecutableNetwork &network) { network_ = network; }

        inline void setRawOutput(bool raw_output) { raw_output_ = raw_output; }

        //getter
        inline const int getMaxBatchSize() const { return max_batch_size_; }

        inline const std::string getModelLoc() const { return model_loc_; }

        inline const std::string getDevice() const { return device_; }

        inline InferenceEngine::ExecutableNetwork &getNetwork() { return network_; }

        inline bool getRawOutput() { return raw_output_; }

        inline bool const getEnable() const { return enable_; }

        virtual const std::string getName() const = 0;

        //exclusive for Read()
        virtual void networkInit(InferenceEngine::CNNNetReader *netReader) = 0;

        virtual void initAndCheckInput(InferenceEngine::CNNNetReader *netReader) = 0;

        virtual void initAndCheckOutput(InferenceEngine::CNNNetReader *net_reader) = 0;

    private:
        mutable bool enablingChecked_ = false;
        mutable bool enable_ = false;
        int max_batch_size_;
        std::string model_loc_;
        std::string device_;
        std::vector<std::string> labels_;
        InferenceEngine::ExecutableNetwork network_;
        InferenceEngine::InferencePlugin *plugin_;
        InferenceEngine::InferRequest::Ptr request_ = nullptr;
        bool raw_output_ = false;
    };

    class FaceDetection : public Detection {
    public:
        struct Result : Detection::Result {
            std::string label;
            float confidence;
            cv::Rect location;
        };

        FaceDetection(const std::string &model_loc, const std::string &device, double show_output_thresh);

        /**
         * @brief This function will add the content of frame into input blob of the network
         * @param frame
         */
        void enqueue(const cv::Mat &frame) override;

        void submitRequest() override;

        void fetchResults() override ;

        inline size_t getResultsLength() override { return results_.size(); }

        inline const Result* getResultPtr(int idx) const override {
            return &results_[idx];
        }

        inline const std::vector<Result> &getAllDetectionResults() const { return results_; }

    protected:
        void networkInit(InferenceEngine::CNNNetReader *net_reader) override;

        void initAndCheckInput(InferenceEngine::CNNNetReader *net_reader) override;

        void initAndCheckOutput(InferenceEngine::CNNNetReader *net_reader) override;

        const std::string getName() const override { return "Face Detection"; };
    private:
        std::vector<Result> results_;
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
}
#endif //SAMPLES_DETECTION_H
