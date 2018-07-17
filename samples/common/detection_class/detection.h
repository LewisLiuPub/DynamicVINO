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

        Detection(const std::string &model_loc, const std::string &device, int max_batch_size);
        /**
         * @brief Read model into network reader, initialize and config the network' s input and output.
         * @return the configured network object
         */
        InferenceEngine::CNNNetwork Read();

        virtual void SubmitRequest();

        virtual void Wait();

        bool Enabled() const;

        void load(InferenceEngine::InferencePlugin &plg);

        inline std::vector<std::string> &GetLabels() { return labels_; }

        void PrintPerformanceCounts();

        virtual void fetchResults() = 0;

    protected:
        //setter
        inline void SetMaxBatchSize(int max_batch_size) { max_batch_size_ = max_batch_size; }

        inline void SetModelLoc(const std::string &model_loc) { model_loc_ = model_loc; }

        inline void SetLabels(const std::vector<std::string> &labels) { labels_ = labels; }

        inline void SetDevice(const std::string &device) { device_ = device; };

        inline void SetRequest(const InferenceEngine::InferRequest::Ptr &request) { request_ = request; };

        inline void SetNetwork(InferenceEngine::ExecutableNetwork &network) { network_ = network; }

        inline void SetRawOutput(bool raw_output) { raw_output_ = raw_output; }

        //getter
        inline const int GetMaxBatchSize() const { return max_batch_size_; }

        inline const std::string &GetModelLoc() const { return model_loc_; }

        inline const std::string &GetDevice() const { return device_; }

        inline InferenceEngine::InferRequest::Ptr &GetRequest() { return request_; }

        inline InferenceEngine::ExecutableNetwork &GetNetwork() { return network_; }

        inline bool GetRawOutput() { return raw_output_; }

        inline bool const GetEnable() const { return enable_; }

        virtual const std::string GetName() const = 0;

        //exclusive for Read()
        virtual void NetworkInit(InferenceEngine::CNNNetReader *netReader) = 0;

        virtual void InitAndCheckInput(InferenceEngine::CNNNetReader *netReader) = 0;

        virtual void InitAndCheckOutput(InferenceEngine::CNNNetReader *net_reader) = 0;

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
        struct Result {
            int label;
            float confidence;
            cv::Rect location;
        };

        FaceDetection(const std::string &model_loc, const std::string &device, double show_output_thresh);

        /**
         * @brief This function will add the content of frame into input blob of the network
         * @param frame
         */
        void Enqueue(const cv::Mat &frame);

        void SubmitRequest() override;

        void fetchResults() override ;

        inline const std::vector<Result> &GetResults() const { return results_; }

    protected:
        void NetworkInit(InferenceEngine::CNNNetReader *net_reader) override;

        void InitAndCheckInput(InferenceEngine::CNNNetReader *net_reader) override;

        void InitAndCheckOutput(InferenceEngine::CNNNetReader *net_reader) override;

        const std::string GetName() const override { return "Face Detection"; };
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
