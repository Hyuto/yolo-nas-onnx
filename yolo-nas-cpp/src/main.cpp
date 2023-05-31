#include <iostream>

#include "cli.hpp"
#include "yolo-nas.hpp"

int main(int argc, char **argv)
{
    Config args = parseCLI(argc, argv);

    cv::Mat img = cv::imread(args.source);
    YoloNAS net(args.netPath, args.imgSize, args.gpu, args.scoreTresh, args.iouTresh);
    net.predict(img);

    cv::imshow(args.source, img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}