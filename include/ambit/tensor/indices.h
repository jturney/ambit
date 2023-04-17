/*
 * @BEGIN LICENSE
 *
 * ambit: C++ library for the implementation of tensor product calculations
 *        through a clean, concise user interface.
 *
 * Copyright (c) 2014-2017 Ambit developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of ambit.
 *
 * Ambit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Ambit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

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
