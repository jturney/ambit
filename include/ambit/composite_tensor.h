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

//
// Created by Justin Turney on 2/4/16.
//

#ifndef AMBIT_COMPOSITE_TENSOR_H
#define AMBIT_COMPOSITE_TENSOR_H

#include <ambit/common_types.h>
#include <ambit/tensor.h>
#include <ambit/blocked_tensor.h>

namespace ambit
{

template <typename TensorType>
class CompositeTensor
{
    string name_;
    vector<TensorType> tensors_;

public:

    CompositeTensor(const string& name, size_t ntensors = 0)
            : name_(name), tensors_(ntensors+1)
    {}

    TensorType& operator()(size_t elem)
    {
        return tensors_[elem];
    }

    const TensorType& operator()(size_t elem) const
    {
        return tensors_[elem];
    }

    CompositeTensor& add(TensorType& newTensor)
    {
        tensors_.push_back(newTensor);
        return *this;
    }
};

}

#endif //AMBIT_COMPOSITE_TENSOR_H
