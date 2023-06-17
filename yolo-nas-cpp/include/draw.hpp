#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class Colors
{
private:
    std::vector<cv::Scalar> palette{cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
                                    cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
                                    cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
                                    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
                                    cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)};
    int n = (int)palette.size();

public:
    inline cv::Scalar get(int i)
    {
        return palette[i % n];
    }
};

void draw_box(cv::Mat &source, cv::Rect &box, std::string &label, float &score, cv::Scalar color, float alpha = 0.25);