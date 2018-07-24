//
// Created by chris on 18-7-20.
//

#include "output.h"

void ImageWindow::feedFrame(const cv::Mat & frame) {
    frame_ = frame;
}

void ImageWindow::prepareData(const DetectionClass::FaceDetection::Result& result){
    std::ostringstream out;
    /**
     * Text Output
     */
    /*out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
        << (ocv_decode_time + ocv_render_time) << " ms";
    cv::putText(frame, out.str(), cv::Point2f(0, 25),
                //cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

    out.str("");
    out << "Face detection time: " << std::fixed << std::setprecision(2)
        << detection.count()
        << " ms ("
        << 1000.f / detection.count() << " fps)";
    cv::putText(frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                cv::Scalar(255, 0, 0));*/
        cv::Rect rect = result.location;

        out.str("");

        {
            out << result.label
                << ": " << std::fixed << std::setprecision(3) << result.confidence;
        }
        cv::putText(frame_,
                    out.str(),
                    cv::Point2f(result.location.x, result.location.y - 15),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.8,
                    cv::Scalar(0, 0, 255));
        cv::rectangle(frame_, result.location, cv::Scalar(100, 100, 100), 1);

}

void ImageWindow::handleOutput() {
    cv::imshow(window_name_, frame_);
}