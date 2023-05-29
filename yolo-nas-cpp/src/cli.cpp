#include <argparse/argparse.hpp>
#include "utils.hpp"
#include "cli.hpp"

Config parseCLI(int argc, char **argv)
{
    argparse::ArgumentParser program("yolo-nas-cpp");
    program.add_description("Detect using YOLO-NAS model");

    program.add_argument("model").help("Path to the YOLO-NAS ONNX model.").metavar("MODEL");
    program.add_argument("-i", "--image").help("Path to the image source");
    program.add_argument("-v", "--video").help("Path to the video source");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::string netPath = program.get<std::string>("model");
    auto imgPath = program.present("-i"),
         vidPath = program.present("-v");

    exists(netPath);
    if (imgPath && vidPath)
    {
        std::cout << LogError("Double Entry", "Please specify either image or video source!") << std::endl;
        std::abort();
    }
    if (!(imgPath || vidPath))
    {
        std::cout << LogError("No Entry", "Please input either image or video source!") << std::endl;
        std::abort();
    }

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

    Config configurations{netPath, type, source};

    return configurations;
}