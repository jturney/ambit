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

#include <mutex> // std::mutex
#include <numeric>

#include <ambit/tensor_pool.h>

namespace ambit
{

std::mutex pool_mutex; // mutex for critical section

TempTensor TensorPool::get_tensor()
{
    // critical section
    pool_mutex.lock();

    Tensor t;
    size_t id;

    // Find a free tensor
    bool found = false;
    for (size_t n = 0; n < tensor_pool_.size(); n++)
    {
        auto &p = tensor_pool_[n];
        // check if the busy flag is false
        if (p.first == false)
        {
            // lock the tensor
            p.first = true;
            // grab the Tensor object and set the id
            t = p.second;
            id = n;
            found = true;
            break;
        }
    }
    // If there is no tensor, create a core tensor with appropriate label and
    // dimensions [1].
    if (not found)
    {
        id = tensor_pool_.size();
        t = Tensor::build(TensorType::CoreTensor,
                          "temporary_tensor_" + std::to_string(id), {1});
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

namespace tensor_pool
{

TensorPool *tensor_pool = nullptr;

void initialize()
{
    if (tensor_pool == nullptr)
    {
        tensor_pool = new TensorPool();
    }
    else
    {
        throw std::runtime_error("tensor_pool::initialize: the TensorPool has "
                                 "already been initialized.");
    }
}

void finalize()
{
    delete tensor_pool;
    tensor_pool = nullptr;
}

TempTensor get_tensor() { return tensor_pool->get_tensor(); }

} // namespace tensor_pool

} // namespace ambit
