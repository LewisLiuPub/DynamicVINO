//
// Created by chris on 18-7-20.
//

#ifndef SAMPLES_OUTPUT_H
#define SAMPLES_OUTPUT_H

#include "detection.h"
#include "opencv2/opencv.hpp"

class BaseOutput {
public:
    virtual void prepareData(const DetectionClass::FaceDetection::Result&) = 0;
    virtual void handleOutput() = 0;
};

class ImageWindow {
public:
    explicit ImageWindow(const std::string &window_name): window_name_(window_name) {
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    }
    void feedFrame(const cv::Mat&);
    void prepareData(const DetectionClass::FaceDetection::Result&);
    void handleOutput();

private:
    const std::string window_name_;
    cv::Mat frame_;
};

#endif //SAMPLES_OUTPUT_H
