#include <iostream>
#include <filesystem>

#include "utils.hpp"

std::string LogInfo(std::string header, std::string body)
{
    return "\033[1m\033[94m" + header + ": \033[0m" + body;
}

std::string LogWarning(std::string header, std::string body)
{
    return "⚠️ \033[1m\033[93m" + header + ": \033[0m" + body;
}

std::string LogError(std::string header, std::string body)
{
    return "❌ \033[1m\033[91m" + header + ": \033[0m" + body;
}

void exists(std::string path)
{
    std::filesystem::path filePath(path);
    if (!std::filesystem::exists(filePath))
    {
        std::cerr << LogError("File Not Found", path) << std::endl;
        std::abort();
    }
}

bool isNumber(const std::string &s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it))
        ++it;
    return !s.empty() && it == s.end();
}

VideoExporter::VideoExporter(cv::VideoCapture &cap, std::string path)
{
    exportPath = path;

    if (exportPath != "")
    {
        if (cap.isOpened())
        {
            int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = cap.get(cv::CAP_PROP_FPS);
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer = cv::VideoWriter(exportPath, fourcc, fps, cv::Size(width, height));
        }
        else
        {
            std::cerr << LogError("Video Exporter", "Opencv Video Capture isn't opened yet!") << std::endl;
            std::abort();
        }
    }
};

void VideoExporter::write(cv::Mat &frame)
{
    if (exportPath != "")
        writer.write(frame);
}

void VideoExporter::close()
{
    if (exportPath != "")
    {
        std::cout << LogInfo("Export Video", exportPath) << std::endl;
        writer.release();
    }
}