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
    void detRescale(cv::Mat &source, cv::Mat &dst, json &metadata);
    void detLongMaxRescale(cv::Mat &source, cv::Mat &dst, json &metadata);
    void padBotRight(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);
    void padCenter(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);
    void _call_fn(std::string name, cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata);

public:
    json prepSteps{{{"DetLongMaxRescale", nullptr}},
                   {{"CenterPad", {{"pad_value", 114}}}},
                   {{"Standardize", {{"max_value", 255.0}}}}};
    cv::Size outShape{640, 640};

    PreProcessing();
    PreProcessing(json &steps, std::vector<int> shape);

    static void rescaleImage(cv::Mat &img, cv::Mat &dst, cv::Size size);
    json run(cv::Mat &img, cv::Mat &dst);
};

class PostProcessing
{
private:
    void rescaleBox(std::vector<float> &box, json &metadata);
    void shiftBox(std::vector<float> &box, json &metadata);
    void _call_fn(std::string name, std::vector<float> &box, json &metadata);

public:
    json prepSteps{{{"DetLongMaxRescale", nullptr}},
                   {{"CenterPad", {{"pad_value", 114}}}},
                   {{"Standardize", {{"max_value", 255.0}}}}};
    float iouThresh = 0.45f;
    float scoreThresh = 0.25f;

    PostProcessing();
    PostProcessing(json &steps, float score, float iou);

    void run(std::vector<std::vector<cv::Mat>> &outputs,
             std::vector<cv::Rect> &boxes,
             std::vector<int> &labels,
             std::vector<float> &scores,
             std::vector<int> &selectedIDX,
             json &metadata);
};