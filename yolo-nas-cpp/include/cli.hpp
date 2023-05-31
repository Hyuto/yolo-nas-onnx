#pragma once

#include <vector>

enum Source
{
    IMAGE,
    VIDEO
};

struct Config
{
    std::string netPath;
    Source type;
    std::string source;
    std::vector<int> imgSize;
    bool gpu;
    float scoreTresh;
    float iouTresh;
};

Config parseCLI(int argc, char **argv);