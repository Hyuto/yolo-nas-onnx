#pragma once

#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum SourceType
{
    IMAGE,
    VIDEO
};

struct Source
{
    SourceType type;
    std::string path;
};

struct Net
{
    std::string path;
    bool gpu;
    std::vector<std::string> labels;
};

struct Processing
{
    std::vector<int> inputShape;
    json PrepSteps;
    float scoreThresh = -1.0f;
    float iouThresh = -1.0f;
};

struct Config
{
    Net net;
    Source source;
    Processing processing;
    std::string exportPath;
};

Config parseCLI(int argc, char **argv);