#include "processing.hpp"
#include "utils.hpp"

#define EXTRACT(x, j) x = j[#x].get<decltype(x)>()

PreProcessing::PreProcessing(json &steps, std::vector<int> shape)
{
    for (auto &step : steps)
        if (!step.is_null())
            prepSteps.push_back(step);

    outShape = cv::Size(shape[0], shape[1]);
}

void PreProcessing::rescaleImage(cv::Mat &img, cv::Mat &dst, cv::Size size)
{
    cv::resize(img, dst, size, 0, 0, cv::INTER_LINEAR);
}

void PreProcessing::standarize(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata)
{
    float max_value;
    EXTRACT(max_value, kwargs);

    dst = (source / max_value);
    metadata.push_back(nullptr);
}

void PreProcessing::normalize(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata)
{
    std::vector<double> mean, std;
    EXTRACT(mean, kwargs);
    EXTRACT(std, kwargs);

    dst = (source - cv::Scalar(mean[0], mean[1], mean[2])) / cv::Scalar(std[0], std[1], std[2]);
    metadata.push_back(nullptr);
}

void PreProcessing::detRescale(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata)
{
    float scaleFactor_h = (float)outShape.height / (float)source.rows,
          scaleFactor_w = (float)outShape.width / (float)source.cols;

    rescaleImage(source, dst, outShape);
    metadata.push_back({{"scale_factors", {scaleFactor_w, scaleFactor_h}}});
}

void PreProcessing::detLongMaxRescale(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata)
{
    float scaleFactor = std::min((float)(outShape.height - 4) / (float)source.rows, (float)(outShape.width - 4) / (float)source.cols);

    if (scaleFactor != 1.0f)
    {
        int newHeight = (int)std::round((float)source.rows * scaleFactor),
            newWidth = (int)std::round((float)source.cols * scaleFactor);
        rescaleImage(source, dst, cv::Size(newWidth, newHeight));
    }
    else
        source.copyTo(dst);

    metadata.push_back({{"scale_factors", {scaleFactor, scaleFactor}}});
}

void PreProcessing::padBotRight(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata)
{
    int padHeight = outShape.height - source.rows,
        padWidth = outShape.width - source.cols;
    int pad_value;
    EXTRACT(pad_value, kwargs);

    cv::copyMakeBorder(source, dst, 0, padHeight, 0, padWidth, cv::BORDER_CONSTANT, cv::Scalar(pad_value, pad_value, pad_value));
    metadata.push_back({{"padding", {0, padHeight, 0, padWidth}}});
}

void PreProcessing::padCenter(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata)
{
    int padHeight = outShape.height - source.rows,
        padWidth = outShape.width - source.cols;
    int padLeft = padWidth / 2,
        padTop = padHeight / 2;

    int pad_value;
    EXTRACT(pad_value, kwargs);

    cv::copyMakeBorder(source, dst, padTop, padHeight - padTop, padLeft, padWidth - padLeft,
                       cv::BORDER_CONSTANT, cv::Scalar(pad_value, pad_value, pad_value));
    metadata.push_back({{"padding", {padTop, padHeight - padTop, padLeft, padWidth - padLeft}}});
}

void PreProcessing::_call_fn(std::string name, cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata)
{
    if (name == "Standardize")
        standarize(source, dst, kwargs, metadata);
    else if (name == "Normalize")
        normalize(source, dst, kwargs, metadata);
    else if (name == "DetRescale")
        detRescale(source, dst, kwargs, metadata);
    else if (name == "DetLongMaxRescale")
        detLongMaxRescale(source, dst, kwargs, metadata);
    else if (name == "BotRightPad")
        padBotRight(source, dst, kwargs, metadata);
    else if (name == "CenterPad")
        padCenter(source, dst, kwargs, metadata);
    else
    {
        std::cerr << LogError("Not Implemented", name + " in preprocessing steps isn't implemented yet!");
        std::abort();
    }
}

json PreProcessing::run(cv::Mat &img, cv::Mat &dst)
{
    cv::cvtColor(img, dst, cv::COLOR_BGR2RGB);
    dst.convertTo(dst, CV_32F);
    json metadata;

    for (auto &step : prepSteps)
        for (auto &[name, kwargs] : step.items())
            _call_fn(name, dst, dst, kwargs, metadata);

    cv::dnn::blobFromImage(dst, dst);

    return metadata;
}

PostProcessing::PostProcessing(json &steps, float score, float iou)
{
    for (auto &step : steps)
        if (!step.is_null())
            prepSteps.push_back(step);
    scoreThresh = score;
    iouThresh = iou;
}

void PostProcessing::rescaleBoxes(cv::Rect &boxes, json &metadata)
{
    std::vector<float> scale_factors;
    EXTRACT(scale_factors, metadata);

    boxes.x /= scale_factors[0];
    boxes.y /= scale_factors[1];
    boxes.width /= scale_factors[0];
    boxes.height /= scale_factors[1];
}

void PostProcessing::shiftBoxes(cv::Rect &boxes, json &metadata)
{
    std::vector<int> padding;
    EXTRACT(padding, metadata);

    boxes.x -= padding[2];
    boxes.y -= padding[0];
    boxes.width -= padding[2];
    boxes.height -= padding[0];
}
