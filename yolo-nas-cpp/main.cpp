#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

class Colors
{
public:
    std::vector<cv::Scalar> palette{cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
                                    cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
                                    cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
                                    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
                                    cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)};
    int n = (int)palette.size();

    cv::Scalar get(int i)
    {
        return palette[i % n];
    }
};

std::vector<std::string> COCO_LABELS{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                                     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                                     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                                     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                                     "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                                     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                     "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                                     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                                     "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

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

void exists(std::filesystem::path &path)
{
    if (!std::filesystem::exists(path))
    {
        std::cerr << "\033[1m\033[91mNot found:\033[0m " + path.generic_string() << std::endl;
        std::abort();
    }
}

int main(int argc, char **argv)
{
    // configs
    std::filesystem::path imgPath{argv[2]};
    std::filesystem::path netPath{argv[1]};
    std::vector<int> netInputShape{1, 3, 640, 640};
    float IOUThresh = 0.45f,
          scoreThresh = 0.3f;
    bool useCUDA = false;
    Colors colors;

    std::string emoji = "🖼️";
    std::cout << emoji + " \033[1m\033[94m" + "Detect: " + "\033[0mmodel=" + netPath.generic_string();
    std::cout << " image=" + imgPath.generic_string() << std::endl;

    exists(imgPath);
    exists(netPath);

    cv::Mat img = cv::imread(imgPath.generic_string());
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

    cv::dnn::Net net = cv::dnn::readNetFromONNX(netPath.generic_string());
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

    cv::imshow(imgPath.filename().generic_string(), img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}