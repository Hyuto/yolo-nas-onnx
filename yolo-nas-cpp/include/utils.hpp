#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

std::string LogInfo(std::string header, std::string body);

std::string LogWarning(std::string header, std::string body);

std::string LogError(std::string header, std::string body);

void exists(std::string path);

const std::vector<std::string> COCO_LABELS{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                                           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                                           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                           "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                                           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                                           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

class VideoExporter
{
private:
    cv::VideoWriter writer;

public:
    std::string exportPath;

    VideoExporter(cv::VideoCapture &cap, std::string path);
    void write(cv::Mat &frame);
    void close();
};