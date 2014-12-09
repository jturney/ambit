#include <tensor/tensor.h>
#include "tensorimpl.h"

namespace tensor {

TensorType Tensor::type() const
{
    return tensor_->type();
}

std::string Tensor::name() const
{
    return tensor_->name();
}

const std::vector<size_t>& Tensor::dims() const
{
    return tensor_->dims();
}

size_t Tensor::rank() const
{
    return tensor_->dims().size();
}

}
