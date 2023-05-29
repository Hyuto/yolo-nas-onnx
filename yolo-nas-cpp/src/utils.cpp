#include <iostream>
#include <filesystem>

std::string BCODE[] = {"\033[95m", "\033[94m", "\033[93m",
                       "\033[91m", "\033[0m", "\033[1m"};

enum BCOLORS
{
    HEADER,
    OKBLUE,
    WARNING,
    FAIL,
    ENDC,
    BOLD
};

inline std::string LogInfo(std::string header, std::string body)
{
    return BCODE[HEADER] + BCODE[OKBLUE] + header + ": " + BCODE[ENDC] + body;
}

inline std::string LogWarning(std::string header, std::string body)
{
    return "⚠️ " + BCODE[HEADER] + BCODE[WARNING] + header + ": " + BCODE[ENDC] + body;
}

inline std::string LogError(std::string header, std::string body)
{
    return "❌ " + BCODE[HEADER] + BCODE[FAIL] + header + ": " + BCODE[ENDC] + body;
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
