#include <iostream>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "cli.hpp"
#include "draw.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    Config args = parseCLI(argc, argv);

    // configs
    std::string imgPath = args.source;
    std::string netPath = args.netPath;
    std::vector<int> netInputShape{1, 3, 640, 640};
    float IOUThresh = 0.45f,
          scoreThresh = 0.3f;
    bool useCUDA = false;
    Colors colors;

    std::string emoji = "🖼️";
    std::cout << emoji + " \033[1m\033[94m" + "Detect: " + "\033[0mmodel=" + netPath;
    std::cout << " image=" + imgPath << std::endl;

    exists(imgPath);
    exists(netPath);

    cv::Mat img = cv::imread(imgPath);
    cv::Mat imgInput;
    cv::cvtColor(img, imgInput, cv::COLOR_BGR2RGB);

    // padding image to [n x n] dim
    int maxSize = std::max(imgInput.cols, imgInput.rows);
    int xPad = maxSize - imgInput.cols,
        yPad = maxSize - imgInput.rows;
    float xRatio = (float)maxSize / (float)netInputShape[3],
          yRatio = (float)maxSize / (float)netInputShape[2];
    cv::copyMakeBorder(imgInput, imgInput, 0, yPad, 0, xPad, cv::BORDER_CONSTANT); // padding black

    imgInput = cv::dnn::blobFromImage(imgInput, 1 / 255.0f, cv::Size(netInputShape[3], netInputShape[2]));

    cv::dnn::Net net = cv::dnn::readNetFromONNX(netPath);
    if (useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        std::cout << "Attempting to use CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    net.setInput(imgInput);
    std::vector<std::vector<cv::Mat>> out;
    std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
    net.forward(out, outNames);
    cv::Mat &rawScores = out[0][0],
            &bboxes = out[1][0];
    rawScores = rawScores.reshape(0, {rawScores.size[1], rawScores.size[2]});
    bboxes = bboxes.reshape(0, {bboxes.size[1], bboxes.size[2]});

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    cv::Mat rowScores;
    for (int i = 0; i < bboxes.size[0]; i++)
    {
        rowScores = rawScores.row(i);
        cv::Point classID;
        double maxScore;
        minMaxLoc(rowScores, 0, &maxScore, 0, &classID);

        if ((float)maxScore < scoreThresh)
            continue;

        int x = (int)(bboxes.at<float>(i, 0) * xRatio),
            y = (int)(bboxes.at<float>(i, 1) * yRatio),
            x1 = (int)(bboxes.at<float>(i, 2) * xRatio),
            y1 = (int)(bboxes.at<float>(i, 3) * yRatio);
        int w = x1 - x,
            h = y1 - y;

        labels.push_back(classID.x);
        scores.push_back((float)maxScore);
        boxes.push_back(cv::Rect(x, y, w, h));
    }
    bboxes.release();
    rawScores.release();
    imgInput.release();
    rowScores.release();

    std::vector<int> selectedIDX;
    cv::dnn::NMSBoxes(boxes, scores, scoreThresh, IOUThresh, selectedIDX);

    for (int x : selectedIDX)
    {
        int classID = labels[x];
        float score = scores[x];
        cv::Scalar color = colors.get(classID);
        cv::rectangle(img, boxes[x], color, 2);
        draw_box(img, boxes[x], classID, score, color);
    }

    cv::imshow(imgPath, img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}