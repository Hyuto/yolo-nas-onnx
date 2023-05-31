#include "yolo-nas.hpp"
#include "utils.hpp"

YoloNAS::YoloNAS(std::string netPath, std::vector<int> imgsz, bool cuda, float score, float iou)
{
    net = cv::dnn::readNetFromONNX(netPath);
    if (cuda && cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        std::cout << LogInfo("Backend", "Attempting to use CUDA") << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    imgSize = imgsz;

    int width = imgSize[0],
        height = imgSize[1];

    netInputShape[3] = width;
    netInputShape[2] = height;
    scoreTresh = score;
    iouTresh = iou;

    warmup(3);
}

void YoloNAS::warmup(int round)
{
    cv::Mat mat(4, netInputShape, CV_32F);
    std::vector<std::vector<cv::Mat>> out;
    for (int i = 0; i < round; i++)
    {
        randu(mat, cv::Scalar(0), cv::Scalar(1));
        net.setInput(mat);
        net.forward(out, net.getUnconnectedOutLayersNames());
    }

    mat.release();
    out[0][0].release();
    out[1][0].release();
}

void YoloNAS::preprocess(cv::Mat &source, cv::Mat &dst, std::vector<float> &ratios)
{
    // padding image to [n x n] dim
    int maxSize = std::max(source.cols, source.rows);
    int xPad = maxSize - source.cols,
        yPad = maxSize - source.rows;
    float xRatio = (float)maxSize / (float)netInputShape[3],
          yRatio = (float)maxSize / (float)netInputShape[2];
    cv::copyMakeBorder(source, dst, 0, yPad, 0, xPad, cv::BORDER_CONSTANT); // padding black

    dst = cv::dnn::blobFromImage(dst, 1.0f / 255.0f, cv::Size(netInputShape[3], netInputShape[2]), cv::Scalar(), true);

    ratios.push_back(xRatio);
    ratios.push_back(yRatio);
}

void YoloNAS::postprocess(std::vector<std::vector<cv::Mat>> &out,
                          std::vector<cv::Rect> &boxes,
                          std::vector<int> &labels,
                          std::vector<float> &scores,
                          std::vector<int> &selectedIDX,
                          std::vector<float> &ratios)
{
    cv::Mat &rawScores = out[0][0],
            &bboxes = out[1][0];
    rawScores = rawScores.reshape(0, {rawScores.size[1], rawScores.size[2]});
    bboxes = bboxes.reshape(0, {bboxes.size[1], bboxes.size[2]});

    cv::Mat rowScores;
    for (int i = 0; i < bboxes.size[0]; i++)
    {
        rowScores = rawScores.row(i);
        cv::Point classID;
        double maxScore;
        minMaxLoc(rowScores, 0, &maxScore, 0, &classID);

        if ((float)maxScore < scoreTresh)
            continue;

        int x = (int)(bboxes.at<float>(i, 0) * ratios[0]),
            y = (int)(bboxes.at<float>(i, 1) * ratios[1]),
            x1 = (int)(bboxes.at<float>(i, 2) * ratios[0]),
            y1 = (int)(bboxes.at<float>(i, 3) * ratios[1]);
        int w = x1 - x,
            h = y1 - y;

        labels.push_back(classID.x);
        scores.push_back((float)maxScore);
        boxes.push_back(cv::Rect(x, y, w, h));
    }
    cv::dnn::NMSBoxes(boxes, scores, scoreTresh, iouTresh, selectedIDX);

    bboxes.release();
    rawScores.release();
    rowScores.release();
}

void YoloNAS::predict(cv::Mat &img)
{
    cv::Mat imgInput;
    std::vector<float> ratios;
    preprocess(img, imgInput, ratios);

    std::vector<std::vector<cv::Mat>> out;
    net.setInput(imgInput);
    net.forward(out, net.getUnconnectedOutLayersNames());
    imgInput.release();

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    std::vector<int> selectedIDX;

    postprocess(out, boxes, labels, scores, selectedIDX, ratios);

    for (int x : selectedIDX)
    {
        int classID = labels[x];
        float score = scores[x];
        cv::Scalar color = colors.get(classID);
        cv::rectangle(img, boxes[x], color, 2);
        draw_box(img, boxes[x], classID, score, color);
    }
}