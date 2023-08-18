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

#include <stdexcept>
#include <ambit/tensor.h>

namespace ambit
{

SlicedTensor::SlicedTensor(Tensor T, const IndexRange &range, double factor)
    : T_(T), range_(range), factor_(factor)
{
    if (T_.rank() != range_.size())
    {
        throw std::runtime_error("Sliced tensor does not have correct number "
                                 "of indices for underlying tensor's rank\n "
                                 "range_ " +
                                 std::to_string(range_.size()) + " rank " +
                                 std::to_string(T.rank()));
    }
    for (size_t ind = 0; ind < T.rank(); ind++)
    {
        if (range_[ind].size() != 2)
            throw std::runtime_error("Each index of an IndexRange should have "
                                     "two elements {start,end+1} in it.");
        if (range_[ind][0] > range_[ind][1])
            throw std::runtime_error(
                "Each index of an IndexRange should end+1>=start in it.");
        if (range_[ind][1] > T_.dims()[ind])
            throw std::runtime_error("IndexRange exceeds size of tensor.");
    }
}

void SlicedTensor::operator=(const SlicedTensor &rhs)
{
    if (T() == rhs.T()) {
        if (range_ == rhs.range_ and factor_ == rhs.factor_) {
            return; // No work to do.
        } else {
            throw std::runtime_error("Non-trivial self-assignment is not allowed.");
        }
    }
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Sliced tensors do not have same rank");
    T_.slice(rhs.T(), range_, rhs.range_, rhs.factor_, 0.0);
}

SlicedTensor& SlicedTensor::operator+=(const SlicedTensor &rhs)
{
    if (T() == rhs.T())
        throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Sliced tensors do not have same rank");
    T_.slice(rhs.T(), range_, rhs.range_, rhs.factor_, 1.0);
    return *this;
}

SlicedTensor& SlicedTensor::operator-=(const SlicedTensor &rhs)
{
    if (T() == rhs.T())
        throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Sliced tensors do not have same rank");
    T_.slice(rhs.T(), range_, rhs.range_, -rhs.factor_, 1.0);
    return *this;
}
}
