#if !defined(TENSOR_INDICES_H)
#define TENSOR_INDICES_H

#include <string>
#include <sstream>
#include <vector>
#include <ambit/tensor.h>

namespace ambit {

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

std::vector<size_t> permutation_order(const std::vector<std::string>& left, const std::vector<std::string>& right);

// => Stuff for contract <= //

int find_index_in_vector(const std::vector<std::string>& vec, const std::string& key);

bool contiguous(const std::vector<std::pair<int, std::string>>& vec);

Dimension permuted_dimension(
    const Dimension& old_dim,
    const std::vector<std::string>& new_order,
    const std::vector<std::string>& old_order);

std::vector<Indices> determine_contraction_result(const LabeledTensor& A, const LabeledTensor& B);

// Returns a comma separated list of the indices
std::string to_string(const Indices& indices);

void print(const Indices &indices);

}

}
#endif
