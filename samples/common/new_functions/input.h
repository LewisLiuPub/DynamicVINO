//
// Created by chris on 18-7-10.
//

#ifndef SAMPLES_BASEINPUTDEVICE_H
#define SAMPLES_BASEINPUTDEVICE_H

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
/**
* @class BaseInputDevice
* @brief This class is an interface for three kinds of input devices: realsense camera, standard camera and video
*/
class BaseInputDevice {
public:
    /**
    * @brief initialize the input device, for cameras, it will turn the camera on and get ready to read frames,
     * for videos, it will open a video file
    * @return whether the input device is successfully turned on
    */
    virtual bool initialize() = 0;
    /**
    * @brief read next frame, and give the value to argument frame
    * @return whether the next frame is successfully read
    */
    virtual bool read(cv::Mat *frame) = 0;
    //TODO
    virtual void config() = 0;
    virtual ~BaseInputDevice() = default;
    inline size_t getWidth() { return width_; }
    inline void setWidth(size_t width) { width_ = width; }
    inline size_t getHeight() { return height_; }
    inline void setHeight(size_t height) { height_ = height; }
    inline bool getIsInit(){ return is_init_; }
    inline void setIsInit(bool is_init) { is_init_ = is_init; }
private:
    size_t width_;
    size_t height_;
    bool is_init_ = false;
};

class RealSenseCamera : public BaseInputDevice {
public:
    bool initialize() override;
    bool read(cv::Mat *frame) override;
    void config() override;
private:
    rs2::config cfg_;
    rs2::pipeline pipe_;
    bool first_read_ = true;
};

class StandardCamera : public BaseInputDevice {
public:
    bool initialize() override;
    bool read(cv::Mat *frame) override;
    void config() override;

private:
    cv::VideoCapture cap;

};

class Video: public BaseInputDevice {
public:
    Video(const std::string&);
    bool initialize() override;
    bool read(cv::Mat *frame) override;
    void config() override;

private:
    cv::VideoCapture cap;
    std::string video_;
};
#endif //SAMPLES_BASEINPUTDEVICE_H
