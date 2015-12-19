#include <ambit/common_types.h>

#include <algorithm>
#include "indices.h"

namespace ambit
{

namespace indices
{

namespace
{

// trim from start
static inline string &ltrim(string &s)
{
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline string &rtrim(string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::not1(std::ptr_fun<int, int>(std::isspace)))
                .base(),
            s.end());
    return s;
}

// trim from both ends
static inline string &trim(string &s) { return ltrim(rtrim(s)); }
}

Indices split(const string &indices)
{
    std::istringstream f(indices);
    string s;
    Indices v;

    if (indices.find(",") != string::npos)
    {
        while (std::getline(f, s, ','))
        {
            string trimmed = trim(s);
            v.push_back(trimmed);
        }
    }
    else
    {
        // simply split the string up
        for (size_t i = 0; i < indices.size(); ++i)
            v.push_back(std::string(1, indices[i]));
    }

    return v;
}

bool equivalent(const Indices &left, const Indices &right)
{
    return left == right;
}

vector<size_t> permutation_order(const Indices &left, const Indices &right)
{
    /// Check that these strings have the same number of indices
    if (left.size() != right.size())
        throw std::runtime_error("Permutation indices not of same rank");

    Indices left2 = left;
    Indices right2 = right;
    std::sort(left2.begin(), left2.end());
    std::sort(right2.begin(), right2.end());

    /// Check that the strings have the same tokens
    for (size_t ind = 0; ind < left2.size(); ind++)
    {
        if (left2[ind] != right2[ind])
            throw std::runtime_error("Permutation indices do not match");
    }

    /// Check that the strings do not have repeats
    for (int ind = 0; ind < ((int)left2.size()) - 1; ind++)
    {
        if (left2[ind] == left2[ind + 1])
            throw std::runtime_error("Permutation indices contain repeats");
    }

    /// Find the indices of the tokens of left in right
    vector<size_t> ret(left.size(), -1);
    for (size_t ind = 0; ind < left.size(); ind++)
    {
        for (size_t ind2 = 0; ind2 < right.size(); ind2++)
        {
            if (left[ind] == right[ind2])
            {
                ret[ind] = ind2;
                break;
            }
        }
    }
    return ret;
}

// => Stuff for contract <= //

int find_index_in_vector(const vector<std::string> &vec,
                         const std::string &key)
{
    for (size_t ind = 0L; ind < vec.size(); ind++)
    {
        if (key == vec[ind])
        {
            return ind;
        }
    }
    return -1;
}
bool contiguous(const vector<std::pair<int, std::string>> &vec)
{
    for (int prim = 0L; prim < ((int)vec.size()) - 1; prim++)
    {
        if (vec[prim + 1].first != vec[prim].first + 1)
        {
            return false;
        }
    }
    return true;
}
Dimension permuted_dimension(const Dimension &old_dim, const Indices &new_order,
                             const Indices &old_order)
{
    vector<size_t> order =
        indices::permutation_order(new_order, old_order);
    Dimension new_dim(order.size(), 0L);
    for (size_t ind = 0L; ind < order.size(); ind++)
    {
        new_dim[ind] = old_dim[order[ind]];
    }
    return new_dim;
}

void print(const Indices &indices)
{
    printf("[ ");
    for (const std::string &index : indices)
    {
        printf("%-4s ", index.c_str());
    }
    printf("]\n");
}

string to_string(const Indices &indices, const std::string &sep)
{
    if (indices.size() == 0)
        return std::string();

    ostringstream ss;

    std::copy(indices.begin(), indices.end() - 1,
              ostream_iterator<string>(ss, sep.c_str()));
    ss << indices.back();

    return ss.str();
}

vector<Indices> determine_contraction_result_from_indices(Indices Aindices,
                                                               Indices Bindices)
{
    vector<Indices> result;

    size_t dimAB = Aindices.size() + Bindices.size();

    std::sort(Aindices.begin(), Aindices.end());
    std::sort(Bindices.begin(), Bindices.end());

    // Find the elements common to A and B
    Indices AB_common(dimAB);
    Indices::iterator it_AB_common;
    it_AB_common = std::set_intersection(Aindices.begin(), Aindices.end(),
                                         Bindices.begin(), Bindices.end(),
                                         AB_common.begin());
    AB_common.resize(it_AB_common - AB_common.begin());
    result.push_back(AB_common);

    // Find the elements in A but not in B
    Indices A_minus_B(dimAB);
    Indices::iterator it_A_minus_B;
    it_A_minus_B =
        std::set_difference(Aindices.begin(), Aindices.end(), Bindices.begin(),
                            Bindices.end(), A_minus_B.begin());
    A_minus_B.resize(it_A_minus_B - A_minus_B.begin());
    result.push_back(A_minus_B);

    // Find the elements in B but not in A
    Indices B_minus_A(dimAB);
    Indices::iterator it_B_minus_A;
    it_B_minus_A =
        std::set_difference(Bindices.begin(), Bindices.end(), Aindices.begin(),
                            Aindices.end(), B_minus_A.begin());
    B_minus_A.resize(it_B_minus_A - B_minus_A.begin());
    result.push_back(B_minus_A);

    Indices AB_unique = A_minus_B;
    AB_unique.insert(AB_unique.end(), B_minus_A.begin(), B_minus_A.end());
    result.push_back(AB_unique);

    return result;
}

vector<Indices> determine_contraction_result(const LabeledTensor &A,
                                             const LabeledTensor &B)
{
    return determine_contraction_result_from_indices(A.indices(), B.indices());
}

} // namespace  indices

} // namespace tensor
