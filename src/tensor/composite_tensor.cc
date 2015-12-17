//
// Created by Justin Turney on 10/21/15.
//

#include <ambit/composite_tensor.h>

namespace ambit
{

CompositeTensor::CompositeTensor() {}

CompositeTensor::CompositeTensor(const CompositeTensor &other)
{
    // Clone the tensors in other into here.
    for (const Tensor &t : other.tensors_)
    {
        tensors_.push_back(t.clone(kCurrent));
    }
}

Tensor &CompositeTensor::add_tensor(Tensor &&new_tensor)
{
    tensors_.emplace_back(std::move(new_tensor));
}
}
