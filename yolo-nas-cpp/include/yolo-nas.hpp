#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "processing.hpp"
#include "draw.hpp"

class YoloNAS
{
private:
    int netInputShape[4] = {1, 3, 0, 0};
    void warmup(int round);
    Colors colors;

public:
    cv::dnn::Net net;
    float scoreThresh;
    float iouThresh;
    std::vector<std::string> classLabels;

    PreProcessing preprocess;
    PostProcessing postprocess;
    YoloNAS(std::string netPath, bool cuda, json &prepSteps, std::vector<int> imgsz, float score, float iou, std::vector<std::string> &labels);
    void predict(cv::Mat &img);
};