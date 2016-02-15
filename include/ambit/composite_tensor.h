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
