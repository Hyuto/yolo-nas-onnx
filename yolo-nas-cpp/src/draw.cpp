#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "utils.hpp"

void draw_box(cv::Mat &source, cv::Rect &box, int &label, float &score, cv::Scalar color, float alpha = 0.25)
{
    // Fill box
    cv::Mat cropBox = source(box);
    cv::Mat colorBox = cv::Mat::ones(cropBox.rows, cropBox.cols, CV_8UC3);
    colorBox = colorBox.setTo(color);
    cv::addWeighted(cropBox, 1 - alpha, colorBox, alpha, 1.0, cropBox);
    cv::rectangle(source, box, color, 2);
    cropBox.release();
    colorBox.release();

    double size = std::min<int>(source.cols, source.rows) * 0.0007;
    int thickness = (int)std::floor((float)std::min<int>(source.cols, source.rows) * 0.001f);
    int baseline = 0;
    std::string text = COCO_LABELS[label] + " - " + std::to_string(score * 100).substr(0, 5) + "%";
    cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, size, thickness, &baseline);
    baseline += thickness;

    cv::Rect textBox((int)(box.x - size), box.y - labelSize.height * 2,
                     (int)(labelSize.width * 1.05), labelSize.height * 2);
    cv::rectangle(source, textBox, color, cv::FILLED);
    cv::putText(source, text, cv::Point(box.x + 1, box.y - (int)(labelSize.height * 0.6)),
                cv::FONT_HERSHEY_SIMPLEX, size, cv::Scalar(255, 255, 255), thickness,
                cv::LINE_AA);
}