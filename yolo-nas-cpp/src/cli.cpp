#include <argparse/argparse.hpp>
#include <fstream>

#include "utils.hpp"
#include "cli.hpp"

Config parseCLI(int argc, char **argv)
{
    argparse::ArgumentParser program("yolo-nas-cpp");
    program.add_description("Detect using YOLO-NAS model");

    program.add_argument("model").help("Path to the YOLO-NAS ONNX model.").metavar("MODEL");
    program.add_argument("-I", "--image").help("Path to the image source").metavar("IMAGE");
    program.add_argument("-V", "--video").help("Path to the video source").metavar("VIDEO");

    program.add_argument("--imgsz")
        .help("Model input size")
        .nargs(1, 2)
        .default_value(std::vector<int>{640, 640})
        .scan<'i', int>();
    program.add_argument("--gpu")
        .default_value(false)
        .implicit_value(true)
        .help("Use GPU if available");
    program.add_argument("--score-thresh")
        .default_value(0.25f)
        .help("Float representing the threshold for deciding when to remove boxes")
        .scan<'g', float>();
    program.add_argument("--iou-thresh")
        .default_value(0.45f)
        .help("Float representing the threshold for deciding whether boxes overlap too much with respect to IOU")
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
    float scoreThresh = program.get<float>("--score-thresh"),
          iouThresh = program.get<float>("--iou-thresh");
    std::vector<int> imgSize = program.get<std::vector<int>>("--imgsz");
    auto imgPath = program.present("-I"),
         vidPath = program.present("-V"),
         customMetadata = program.present("--custom-metadata"),
         exportArgs = program.present("--export");

    if (imgPath && vidPath)
    {
        std::cerr << LogError("Double Entry", "Please specify either image or video source!") << std::endl;
        std::abort();
    }
    else if (!(imgPath || vidPath))
    {
        std::cerr << LogError("No Entry", "Please input either image or video source!") << std::endl;
        std::abort();
    }

    Source source;
    if (imgPath)
    {
        exists(imgPath.value());
        source.type = IMAGE;
        source.path = imgPath.value();
    }
    else if (vidPath)
    {
        exists(vidPath.value());
        source.type = VIDEO;
        source.path = vidPath.value();
    }

    if (imgSize.size() == 1)
        imgSize.push_back(imgSize[0]);

    if (customMetadata)
    {
        exists(customMetadata.value());

        /* std::ifstream f(customMetadata.value());
        json data = json::parse(f);

        for (auto x : data["prep_steps"])
        {
            if (!x.is_null())
                std::cout << x.dump(4) << std::endl;
        } */
    }

    json prepsteps;

    exists(netPath);
    Net net{netPath, useGPU, COCO_LABELS};
    Processing processing{imgSize, prepsteps, scoreThresh, iouThresh};

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
    if (customMetadata)
        std::cout << " custom-metadata=" << customMetadata.value();
    if (exportArgs)
        std::cout << " export=" << exportPath;
    std::cout << std::endl;

    return configurations;
}