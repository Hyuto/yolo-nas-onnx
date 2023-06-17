#include <argparse/argparse.hpp>
#include <fstream>

#include "utils.hpp"
#include "cli.hpp"

#define EXTRACT(x, j) x = j[#x].get<decltype(x)>()

Config parseCLI(int argc, char **argv)
{
    argparse::ArgumentParser program("yolo-nas-cpp");
    program.add_description("Detect using YOLO-NAS model");

    program.add_argument("model").help("Path to the YOLO-NAS ONNX model.").metavar("MODEL");
    program.add_argument("-I", "--image").help("Path to the image source").metavar("IMAGE");
    program.add_argument("-V", "--video").help("Path to the video source").metavar("VIDEO");

    program.add_argument("--imgsz")
        .help("Model input size [default: {640 640}]")
        .nargs(1, 2)
        .scan<'i', int>();
    program.add_argument("--gpu")
        .default_value(false)
        .implicit_value(true)
        .help("Use GPU if available");
    program.add_argument("--score-thresh")
        .help("Float representing the threshold for deciding when to remove boxes [default: 0.25]")
        .scan<'g', float>();
    program.add_argument("--iou-thresh")
        .help("Float representing the threshold for deciding whether boxes overlap too much with respect to IOU [default: 0.45]")
        .scan<'g', float>();

    program.add_argument("--export")
        .help("Export to a file (path with extension | mp4 is a must for video)");
    program.add_argument("--custom-metadata")
        .help("Path to metadata file (Generated from https://gist.github.com/Hyuto/f3db1c0c2c36308284e101f441c2555f)");

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
    auto imgPathArgs = program.present<std::string>("-I"),
         vidPathArgs = program.present<std::string>("-V"),
         customMetadataArgs = program.present<std::string>("--custom-metadata"),
         exportArgs = program.present<std::string>("--export");
    auto scoreThreshArgs = program.present<float>("--score-thresh"),
         iouThreshArgs = program.present<float>("--iou-thresh");
    auto imgSizeArgs = program.present<std::vector<int>>("--imgsz");

    if (imgPathArgs && vidPathArgs)
    {
        std::cerr << LogError("Double Entry", "Please specify either image or video source!") << std::endl;
        std::abort();
    }
    else if (!(imgPathArgs || vidPathArgs))
    {
        std::cerr << LogError("No Entry", "Please input either image or video source!") << std::endl;
        std::abort();
    }

    Source source;
    if (imgPathArgs)
    {
        exists(imgPathArgs.value());
        source.type = IMAGE;
        source.path = imgPathArgs.value();
    }
    else if (vidPathArgs)
    {
        std::string vidPath = vidPathArgs.value();
        if (vidPath != "0")
            exists(vidPath);
        source.type = VIDEO;
        source.path = vidPath;
    }

    Processing processing;
    Net net;
    if (customMetadataArgs)
    {
        exists(customMetadataArgs.value());

        std::ifstream f(customMetadataArgs.value());
        json metadata = json::parse(f);

        std::vector<int> original_insz;
        EXTRACT(original_insz, metadata);
        processing.inputShape = {original_insz[3], original_insz[2]};

        float iou_thres, score_thres;
        EXTRACT(iou_thres, metadata);
        EXTRACT(score_thres, metadata);
        processing.iouThresh = iouThreshArgs ? iouThreshArgs.value() : iou_thres;
        processing.scoreThresh = scoreThreshArgs ? iouThreshArgs.value() : score_thres;

        processing.PrepSteps = metadata["prep_steps"];

        std::vector<std::string> labels;
        EXTRACT(labels, metadata);
        net.labels = labels;
    }

    if (imgSizeArgs)
    {
        std::vector<int> imgsz = imgSizeArgs.value();
        if (imgsz.size() == 1)
            imgsz.push_back(imgsz[0]);

        if (processing.inputShape.size() == 2)
            if (!(imgsz == processing.inputShape))
                std::cout << LogWarning("Input Size", "Input size is different from Original Input size from metadata. This will lead to low detection performance or Runtime Error!") << std::endl;
        processing.inputShape = imgsz;
    }
    else if (processing.inputShape.size() == 0)
        processing.inputShape = {640, 640};

    if (processing.scoreThresh == -1.0f)
        processing.scoreThresh = scoreThreshArgs ? scoreThreshArgs.value() : 0.25f;
    if (processing.iouThresh == -1.0f)
        processing.iouThresh = iouThreshArgs ? iouThreshArgs.value() : 0.45f;

    exists(netPath);
    net.path = netPath;
    net.gpu = useGPU;
    if (net.labels.size() == 0)
        net.labels = COCO_LABELS;

    std::string exportPath = exportArgs ? exportArgs.value() : "";

    Config configurations{net, source, processing, exportPath};

    std::string emoji = configurations.source.type == IMAGE ? "🖼️" : "📷";
    std::cout << emoji + LogInfo(" Detect", "model=" + configurations.net.path);
    std::cout << " source=" + configurations.source.path;
    std::cout << " imgsz="
              << "[" << configurations.processing.inputShape[0] << "," << configurations.processing.inputShape[1] << "]";
    std::cout << " gpu=" << (configurations.net.gpu ? "true" : "false");
    std::cout << " score-thresh=" << configurations.processing.scoreThresh;
    std::cout << " iou-thresh=" << configurations.processing.iouThresh;
    if (customMetadataArgs)
        std::cout << " custom-metadata=" << customMetadataArgs.value();
    if (exportArgs)
        std::cout << " export=" << exportPath;
    std::cout << std::endl;

    return configurations;
}