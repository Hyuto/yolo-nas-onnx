#pragma once

#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

class PreProcessing
{
private:
    void standarize(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);
    void normalize(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);
    void detRescale(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);
    void detLongMaxRescale(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);
    void padBotRight(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);
    void padCenter(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);
    void _call_fn(std::string name, cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);

public:
    cv::Size outShape;
    json prepSteps;
    PreProcessing(json &steps, std::vector<int> shape);
    static void rescaleImage(cv::Mat &img, cv::Mat &dst, cv::Size size);
    json run(cv::Mat &img, cv::Mat &dst);
};

class PostProcessing
{
private:
    void rescaleBoxes(cv::Rect &boxes, json &metadata);
    void shiftBoxes(cv::Rect &boxes, json &metadata);

public:
    float iouThresh;
    float scoreThresh;
    json prepSteps;
    PostProcessing(json &steps, float score, float iou);
    void run(std::vector<std::vector<cv::Mat>> &outputs,
             std::vector<cv::Rect> &boxes,
             std::vector<int> &labels,
             std::vector<float> &scores,
             std::vector<int> &selectedIDX,
             json &metadata);
};