#include "indices.h"
#include <algorithm>

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

std::vector<int> permutation_order(const std::vector<std::string>& left, const std::vector<std::string>& right)
{
    /// Check that these strings have the same number of indices
    if (left.size() != right.size()) throw std::runtime_error("Permutation indices not of same rank");

    std::vector<std::string> left2 = left;
    std::vector<std::string> right2 = right;
    std::sort(left2.begin(),left2.end());
    std::sort(right2.begin(),right2.end());

    /// Check that the strings have the same tokens
    for (int ind = 0; ind < left2.size(); ind++) {
        if (left2[ind] != right2[ind]) throw std::runtime_error("Permutation indices do not match");
    }

    /// Check that the strings do not have repeats
    for (int ind = 0; ind < ((int)left2.size()) - 1; ind++) {
        if (left2[ind] == left2[ind+1]) throw std::runtime_error("Permutation indices contain repeats");
    }

    /// Find the indices of the tokens of left in right
    std::vector<int> ret(left.size(),-1);
    for (int ind = 0; ind < left.size(); ind++) {
        for (int ind2 = 0; ind2 < right.size(); ind2++) {
            if (left[ind] == right[ind2]) {
                ret[ind] = ind2;
                break;
            }
        } 
    } 
    return ret;
}

// => Stuff for contract <= //

int find_index_in_vector(const std::vector<std::string>& vec, const std::string& key)
{
    for (size_t ind = 0L; ind < vec.size(); ind++) {
        if (key == vec[ind]) {
            return ind;
        }
    } 
    return -1;
}
bool contiguous(const std::vector<std::pair<int, std::string>>& vec) 
{
    for (int prim = 0L; prim < ((int)vec.size()) - 1; prim++) {
        if (vec[prim+1].first != vec[prim].first + 1) {
            return false;
        }
    }
    return true;
}
Dimension permuted_dimension(
    const Dimension& old_dim, 
    const std::vector<std::string>& new_order,
    const std::vector<std::string>& old_order)
{
    std::vector<int> order = indices::permutation_order(new_order,old_order);
    Dimension new_dim(order.size(),0L);
    for (size_t ind = 0L; ind < order.size(); ind++) {
        new_dim[ind] = old_dim[order[ind]];
    }
    return new_dim;
}

}

}
