#include <opencv2/dnn.hpp>

#include "processing.hpp"
#include "utils.hpp"

#define EXTRACT(x, j) x = j[#x].get<decltype(x)>()

PreProcessing::PreProcessing() {}

PreProcessing::PreProcessing(json &steps, std::vector<int> shape)
{
    if (!steps.is_null())
    {
        prepSteps = {};
        for (auto &step : steps)
            if (!step.is_null())
                prepSteps.push_back(step);
    }

    outShape = cv::Size(shape[0], shape[1]);
}

void PreProcessing::rescaleImage(cv::Mat &img, cv::Mat &dst, cv::Size size)
{
    cv::resize(img, dst, size, 0, 0, cv::INTER_LINEAR);
}

void PreProcessing::standarize(cv::Mat &source, cv::Mat &dst, json &kwargs, json &metadata)
{
    double max_value;
    EXTRACT(max_value, kwargs);

    source.convertTo(dst, CV_32F, 1 / max_value);
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

void PreProcessing::detRescale(cv::Mat &source, cv::Mat &dst, json &metadata)
{
    float scaleFactor_h = (float)outShape.height / (float)source.rows,
          scaleFactor_w = (float)outShape.width / (float)source.cols;

    rescaleImage(source, dst, outShape);
    metadata.push_back({{"scale_factors", {scaleFactor_w, scaleFactor_h}}});
}

void PreProcessing::detLongMaxRescale(cv::Mat &source, cv::Mat &dst, json &metadata)
{
    float scaleFactor = std::min((float)(outShape.height - 4) / (float)source.rows,
                                 (float)(outShape.width - 4) / (float)source.cols);

    if (scaleFactor != 1.0f)
    {
        int newHeight = (int)std::round((float)source.rows * scaleFactor),
            newWidth = (int)std::round((float)source.cols * scaleFactor);
        rescaleImage(source, dst, cv::Size(newWidth, newHeight));
    }
    else
        dst = source;

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
        detRescale(source, dst, metadata);
    else if (name == "DetLongMaxRescale")
        detLongMaxRescale(source, dst, metadata);
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
    img.copyTo(dst);

    json metadata;
    for (auto &step : prepSteps)
        for (auto &[name, kwargs] : step.items())
            _call_fn(name, dst, dst, kwargs, metadata);

    cv::dnn::blobFromImage(dst, dst, 1, cv::Size(), cv::Scalar(), true, false);

    return metadata;
}

PostProcessing::PostProcessing() {}

PostProcessing::PostProcessing(json &steps, float score, float iou)
{
    if (!steps.is_null())
    {
        prepSteps = {};
        for (auto &step : steps)
            if (!step.is_null())
                prepSteps.push_back(step);
    }

    scoreThresh = score;
    iouThresh = iou;
}

void PostProcessing::rescaleBox(std::vector<float> &box, json &metadata)
{
    std::vector<float> scale_factors;
    EXTRACT(scale_factors, metadata);

    box[0] /= scale_factors[0];
    box[1] /= scale_factors[1];
    box[2] /= scale_factors[0];
    box[3] /= scale_factors[1];
}

void PostProcessing::shiftBox(std::vector<float> &box, json &metadata)
{
    std::vector<float> padding;
    EXTRACT(padding, metadata);

    box[0] -= padding[2];
    box[1] -= padding[0];
    box[2] -= padding[2];
    box[3] -= padding[0];
}

void PostProcessing::_call_fn(std::string name, std::vector<float> &box, json &metadata)
{
    if (name == "DetRescale")
        rescaleBox(box, metadata);
    else if (name == "DetLongMaxRescale")
        rescaleBox(box, metadata);
    else if (name == "BotRightPad")
        shiftBox(box, metadata);
    else if (name == "CenterPad")
        shiftBox(box, metadata);
    else if (name == "Standardize" || name == "Normalize")
        ;
    else
    {
        std::cerr << LogError("Not Implemented", name + " in postprocessing steps isn't implemented yet!");
        std::abort();
    }
}

void PostProcessing::run(std::vector<std::vector<cv::Mat>> &outputs,
                         std::vector<cv::Rect> &boxes,
                         std::vector<int> &labels,
                         std::vector<float> &scores,
                         std::vector<int> &selectedIDX,
                         json &metadata)
{
    cv::Mat &rawScores = outputs[0][0],
            &bboxes = outputs[1][0];
    rawScores = rawScores.reshape(0, {rawScores.size[1], rawScores.size[2]});
    bboxes = bboxes.reshape(0, {bboxes.size[1], bboxes.size[2]});

    cv::Mat rowScores;
    for (int i = 0; i < bboxes.size[0]; i++)
    {
        rowScores = rawScores.row(i);
        cv::Point classID;
        double maxScore;
        minMaxLoc(rowScores, 0, &maxScore, 0, &classID);

        if ((float)maxScore < scoreThresh)
            continue;

        std::vector<float> box{bboxes.at<float>(i, 0),  // x
                               bboxes.at<float>(i, 1),  // y
                               bboxes.at<float>(i, 2),  // x1
                               bboxes.at<float>(i, 3)}; // y1

        size_t idx = prepSteps.size();
        for (json::reverse_iterator step = prepSteps.rbegin(); step != prepSteps.rend(); ++step)
        {
            idx--;
            for (auto &e : (*step).items())
                _call_fn(e.key(), box, metadata[idx]);
        }

        labels.push_back(classID.x);
        scores.push_back((float)maxScore);
        boxes.push_back(cv::Rect((int)box[0], (int)box[1], (int)(box[2] - box[0]), (int)(box[3] - box[1])));
    }
    cv::dnn::NMSBoxes(boxes, scores, scoreThresh, iouThresh, selectedIDX);

    bboxes.release();
    rawScores.release();
    rowScores.release();
}