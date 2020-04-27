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
 * You should have received a copy of the GNU Lesser General Public License
 * along with ambit; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

//#include <stdexcept>
//#include <string>
//#include <vector>

//#include "macros.h"

#include <mutex> // std::mutex
#include <numeric>

#include <ambit/tensor_pool.h>

//#if defined(__clang__)
//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wunused-parameter"
//#endif

namespace ambit
{

std::mutex pool_mutex; // mutex for critical section

TempTensor TensorPool::get_tensor(TensorType type, const string &name,
                                  const Dimension &dims)
{
    // critical section
    pool_mutex.lock();

    size_t target_numel =
        std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1),
                        std::multiplies<size_t>());
    Tensor t;
    size_t id;

    // Find a free tensor of the appropriate size
    bool found = false;
    for (size_t n = 0; n < tensor_pool_.size(); n++)
    {
        auto &p = tensor_pool_[n];
        // check if the busy flag is false
        if (p.first == false)
        {
            // lock the tensor
            p.first = true;
            // resize the tensor
            p.second.resize(dims);
            // grab the Tensor object and set the id
            t = p.second;
            id = n;
            found = true;
            break;
        }
    }
    // If there is no tensor, create one
    if (not found)
    {
        t = Tensor::build(type, name, dims);
        id = tensor_pool_.size();
        tensor_pool_.push_back(std::make_pair(true, t));
    }
    pool_mutex.unlock();
    return TempTensor(id, t, this);
}

void TensorPool::release_tensor(size_t id)
{
    // critical section
    pool_mutex.lock();
    tensor_pool_[id].first = false;
    pool_mutex.unlock();
}

void TensorPool::reset() { tensor_pool_.clear(); }
} // namespace ambit
