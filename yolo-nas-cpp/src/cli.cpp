#include <argparse/argparse.hpp>
#include "utils.hpp"
#include "cli.hpp"

Config parseCLI(int argc, char **argv)
{
    argparse::ArgumentParser program("yolo-nas-cpp");
    program.add_description("Detect using YOLO-NAS model");

    program.add_argument("model").help("Path to the YOLO-NAS ONNX model.").metavar("MODEL");
    program.add_argument("-i", "--image").help("Path to the image source").metavar("IMAGE");
    program.add_argument("-v", "--video").help("Path to the video source").metavar("VIDEO");

    program.add_argument("--imgsz")
        .help("Model input size")
        .nargs(1, 2)
        .default_value(std::vector<int>{640, 640})
        .scan<'i', int>();
    program.add_argument("--gpu")
        .default_value(false)
        .implicit_value(true)
        .help("Use GPU if available");
    program.add_argument("--score-tresh")
        .default_value(0.25f)
        .help("Float representing the threshold for deciding when to remove boxes")
        .scan<'g', float>();
    program.add_argument("--iou-tresh")
        .default_value(0.45f)
        .help("Float representing the threshold for deciding whether boxes overlap too much with respect to IOU")
        .scan<'g', float>();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << LogError("Parser Error", err.what()) << std::endl;
        std::cerr << program;
        std::abort();
    }

    std::string netPath = program.get<std::string>("model");
    bool useGPU = program.get<bool>("--gpu");
    float scoreTresh = program.get<float>("--score-tresh"),
          iouTresh = program.get<float>("--iou-tresh");
    std::vector<int> imgSize = program.get<std::vector<int>>("--imgsz");
    auto imgPath = program.present("-i"),
         vidPath = program.present("-v");

    exists(netPath);
    if (imgPath && vidPath)
    {
        std::cerr << LogError("Double Entry", "Please specify either image or video source!") << std::endl;
        std::abort();
    }
    if (!(imgPath || vidPath))
    {
        std::cerr << LogError("No Entry", "Please input either image or video source!") << std::endl;
        std::abort();
    }
    if (imgSize.size() == 1)
        imgSize.push_back(imgSize[0]);

    Source type;
    std::string source;
    if (imgPath)
    {
        exists(imgPath.value());
        type = IMAGE;
        source = imgPath.value();
    }
    else if (vidPath)
    {
        exists(vidPath.value());
        type = VIDEO;
        source = vidPath.value();
    }

    Config configurations{netPath, type, source, imgSize, useGPU, scoreTresh, iouTresh};

    std::string emoji = configurations.type == IMAGE ? "🖼️" : "📷";
    std::cout << emoji + LogInfo(" Detect", "model=" + configurations.netPath);
    std::cout << " source=" + configurations.source;
    std::cout << " imgsz="
              << "[" << configurations.imgSize[0] << "," << configurations.imgSize[1] << "]";
    std::cout << " gpu=" << (configurations.gpu ? "true" : "false");
    std::cout << " score-tresh=" << configurations.scoreTresh;
    std::cout << " iou-tresh=" << configurations.iouTresh << std::endl;

    return configurations;
}