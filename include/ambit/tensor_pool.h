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

#if !defined(TENSOR_POOL_H)
#define TENSOR_POOL_H

#include <vector>

#include <ambit/tensor.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

namespace ambit
{

class TempTensor;

class TensorPool
{
  public:
    // => Constructors <= //

    /**
     * Returns a temporary Core tensor. This function is thread safe.
     *
     * Results:
     *  @return a TempTensor object that will automatically make the
     *  tensor available to the pool upon destruction.
     **/
    TempTensor get_tensor();

    /**
     * Release a temporary Tensor
     *
     * Parameters:
     *  @param id the id of the Tensor
     *
     * Results:
     *  release a temporary tensor and make it available for a new call from
     *  get_tensor
     **/
    void release_tensor(size_t id);

    /**
     * Frees the TensorPool's internal memory allocation.
     */
    void reset();

  private:
    std::vector<std::pair<bool, Tensor>> tensor_pool_;
};

class TempTensor
{
  public:
    // => Constructor <= //
    TempTensor(size_t id, Tensor t, TensorPool *tp) : id_(id), t_(t), tp_(tp) {}
    ~TempTensor() { tp_->release_tensor(id_); }
    Tensor tensor() { return t_; }

  private:
    size_t id_;
    Tensor t_;
    TensorPool *tp_;
};

namespace tensor_pool
{
void initialize();
void finalize();
TempTensor get_tensor();
} // namespace tensor_pool
} // namespace ambit

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
