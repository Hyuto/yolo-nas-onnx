#include <iostream>

#include "utils.hpp"
#include "cli.hpp"
#include "yolo-nas.hpp"

int main(int argc, char **argv)
{
    Config args = parseCLI(argc, argv);
    YoloNAS net(args.net.path, args.net.gpu, args.processing.PrepSteps,
                args.processing.inputShape, args.processing.scoreThresh, args.processing.iouThresh);

    if (args.source.type == IMAGE)
    {
        cv::Mat img = cv::imread(args.source.path);
        net.predict(img);

        cv::namedWindow(args.source.path, cv::WINDOW_NORMAL);
        cv::imshow(args.source.path, img);
        cv::waitKey(0);

        if (args.exportPath != "")
        {
            std::cout << LogInfo("Export Image", args.exportPath) << std::endl;
            cv::imwrite(args.exportPath, img);
        }
    }
    else if (args.source.type == VIDEO)
    {
        cv::VideoCapture cap(args.source.path == "0" ? 0 : args.source.path);
        std::string name = args.source.path == "0" ? "Webcam" : args.source.path;

        // check if camera opened successfully
        if (!cap.isOpened())
        {
            std::cerr << LogError("Video Capture", "Error opening video stream or file") << std::endl;
            std::abort();
        }

        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        std::cout << LogInfo("Processing video", "press 'q' to exit.") << std::endl;

        VideoExporter writer(cap, args.exportPath);
        cv::Mat frame;
        while (true)
        {
            cap >> frame;

            if (frame.empty())
                break;

            net.predict(frame);
            cv::imshow(name, frame);
            writer.write(frame);

            char c = (char)cv::waitKey(1);
            if (c == 113)
                break;
        }
        cap.release();
        writer.close();
    }
    cv::destroyAllWindows();

    return 0;
}