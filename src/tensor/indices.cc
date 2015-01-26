#include "indices.h"

namespace tensor {

namespace indices {

namespace {

// trim from start
static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}

}

std::vector<std::string> split(const std::string &indices)
{
    std::istringstream f(indices);
    std::string s;
    std::vector<std::string> v;

    if (indices.find(",") != std::string::npos) {
        while (std::getline(f, s, ',')) {
            std::string trimmed = trim(s);
            v.push_back(trimmed);
        }
    }
    else {
        // simply split the string up
        for (size_t i = 0; i < indices.size(); ++i)
            v.push_back(std::string(1, indices[i]));
    }

    return v;
}

bool equivalent(const std::vector<std::string> &left, const std::vector<std::string> &right)
{
    return left == right;
}

}

}
