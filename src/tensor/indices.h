#if !defined(TENSOR_INDICES_H)
#define TENSOR_INDICES_H

#include <string>
#include <sstream>
#include <vector>

namespace tensor {

namespace indices {

/** Takes a string of indices and splits them into a vector of strings.
 *
 * If a comma is found in indices then they are split on the comma.
 * If no comma is found it assumes the indices are one character in length.
 *
 */
std::vector<std::string> split(const std::string& indices);

/** Returns true if the two indices are equivalent. */
bool equivalent(const std::vector<std::string>& left, const std::vector<std::string>& right);

std::vector<int> permutation_order(const std::vector<std::string>& left, const std::vector<std::string>& right);

}

}

#endif
