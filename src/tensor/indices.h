#if !defined(TENSOR_INDICES_H)
#define TENSOR_INDICES_H

#include <string>
#include <sstream>
#include <vector>
#include <ambit/tensor.h>

namespace ambit
{

namespace indices
{

/** Takes a string of indices and splits them into a vector of strings.
 *
 * If a comma is found in indices then they are split on the comma.
 * If no comma is found it assumes the indices are one character in length.
 *
 */
Indices split(const string &indices);

/** Returns true if the two indices are equivalent. */
bool equivalent(const Indices &left, const Indices &right);

vector<size_t> permutation_order(const Indices &left, const Indices &right);

// => Stuff for contract <= //

int find_index_in_vector(const Indices &vec, const string &key);

bool contiguous(const vector<pair<int, string>> &vec);

Dimension permuted_dimension(const Dimension &old_dim, const Indices &new_order,
                             const Indices &old_order);

vector<Indices> determine_contraction_result(const LabeledTensor &A,
                                             const LabeledTensor &B);
vector<Indices> determine_contraction_result_from_indices(Indices Aindices,
                                                          Indices Bindices);

// Returns a comma separated list of the indices
string to_string(const Indices &indices, const string &sep = ",");

void print(const Indices &indices);
}
}
#endif
