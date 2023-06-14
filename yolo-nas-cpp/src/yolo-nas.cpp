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

PreprocessingMetadata YoloNAS::preprocess(cv::Mat &source, cv::Mat &dst)
{
    float height = (float)source.rows,
          width = (float)source.cols;
    float scaleFactor = std::min((float)(netInputShape[2] - 4) / height, (float)(netInputShape[3] - 4) / width);

    if (scaleFactor != 1.0f)
    {
        int new_height = (int)std::round(height * scaleFactor),
            new_width = (int)std::round(width * scaleFactor);
        cv::resize(source, dst, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    }

    // padding image to [n x n] dim
    int xPad = netInputShape[3] - dst.cols,
        yPad = netInputShape[2] - dst.rows;
    int padTop = (int)std::floor(yPad / 2),
        padLeft = (int)std::floor(xPad / 2);
    std::vector<int> padding{padTop, yPad - padTop, padLeft, xPad - padLeft};
    cv::copyMakeBorder(dst, dst, padding[0], padding[1], padding[2], padding[3], cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)); // padding black

    dst.convertTo(dst, CV_32F, 1 / 255.0);
    dst = cv::dnn::blobFromImage(dst, 1, cv::Size(), cv::Scalar(), true, false);

    PreprocessingMetadata metadata{scaleFactor, padding};
    return metadata;
}

void YoloNAS::postprocess(std::vector<std::vector<cv::Mat>> &out,
                          std::vector<cv::Rect> &boxes,
                          std::vector<int> &labels,
                          std::vector<float> &scores,
                          std::vector<int> &selectedIDX,
                          PreprocessingMetadata &metadata)
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

        int x = (int)((bboxes.at<float>(i, 0) - (float)metadata.padding[2]) / metadata.scaleFactor),
            y = (int)((bboxes.at<float>(i, 1) - (float)metadata.padding[0]) / metadata.scaleFactor),
            x1 = (int)((bboxes.at<float>(i, 2) - (float)metadata.padding[2]) / metadata.scaleFactor),
            y1 = (int)((bboxes.at<float>(i, 3) - (float)metadata.padding[0]) / metadata.scaleFactor);
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
    PreprocessingMetadata metadata = preprocess(img, imgInput);

    std::vector<std::vector<cv::Mat>> out;
    net.setInput(imgInput);
    net.forward(out, net.getUnconnectedOutLayersNames());
    imgInput.release();

    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels, selectedIDX;

    postprocess(out, boxes, labels, scores, selectedIDX, metadata);

    for (int x : selectedIDX)
    {
        int classID = labels[x];
        float score = scores[x];
        cv::Scalar color = colors.get(classID);
        cv::rectangle(img, boxes[x], color, 2);
        draw_box(img, boxes[x], classID, score, color);
    }
}