#include <iostream>
#include <filesystem>

std::string BCODE[] = {"\033[94m", "\033[93m", "\033[91m", "\033[0m", "\033[1m"};

enum BCOLORS
{
    OKBLUE,
    WARNING,
    FAIL,
    ENDC,
    BOLD
};

std::string LogInfo(std::string header, std::string body)
{
    return BCODE[BOLD] + BCODE[OKBLUE] + header + ": " + BCODE[ENDC] + body;
}

std::string LogWarning(std::string header, std::string body)
{
    return "⚠️ " + BCODE[BOLD] + BCODE[WARNING] + header + ": " + BCODE[ENDC] + body;
}

std::string LogError(std::string header, std::string body)
{
    return "❌ " + BCODE[BOLD] + BCODE[FAIL] + header + ": " + BCODE[ENDC] + body;
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
