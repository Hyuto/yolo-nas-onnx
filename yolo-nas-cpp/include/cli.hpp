#pragma once

#include <optional>

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
};

Config parseCLI(int argc, char **argv);