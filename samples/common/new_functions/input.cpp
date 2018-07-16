//
// Created by chris on 18-7-10.
//

#include "input.h"

bool RealSenseCamera::initialize() {
    cfg_.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    setIsInit(pipe_.start(cfg_));
    setWidth(640);
    setHeight(480);
    return bool(getIsInit());
}

bool RealSenseCamera::read(cv::Mat *frame) {
    if (!getIsInit()) { return false; }
    if (first_read_) {
        rs2::frameset frames;
        for(int i = 0; i < 30; i++)
        {
            //Wait for all configured streams to produce a frame
            try {
                frames = pipe_.wait_for_frames();
            } catch (...) {
                return false;
            }
        }
        first_read_ = false;
    }
    rs2::frameset data = pipe_.wait_for_frames(); // Wait for next set of frames from the camera
    rs2::frame color_frame;
    try {
        color_frame = data.get_color_frame();
    } catch (...) {
        return false;
    }
    cv::Mat(cv::Size((int) getWidth(), (int) getHeight()), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP).copyTo(*frame);
    return true;
}

void RealSenseCamera::config() {
    //TODO
}


bool StandardCamera::initialize() {
    setIsInit(cap.open(0));
    setWidth((size_t)cap.get(CV_CAP_PROP_FRAME_WIDTH));
    setHeight((size_t)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    return getIsInit();
}

bool StandardCamera::read(cv::Mat *frame) {
    if (!getIsInit()) { return false; }
    cap.grab();
    return cap.retrieve(*frame);
}

void StandardCamera::config() {
    //TODO
}

Video::Video(const std::string & video) {
    video_.assign(video);
}
bool Video::initialize() {
    setIsInit(cap.open(video_));
    setWidth((size_t)cap.get(CV_CAP_PROP_FRAME_WIDTH));
    setHeight((size_t)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    return getIsInit();
}
bool Video::read(cv::Mat *frame) {
    if (!getIsInit()) { return false; }
    cap.grab();
    return cap.retrieve(*frame);
}
void Video::config() {
    //TODO
}
