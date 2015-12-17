//
// Created by Justin Turney on 10/19/15.
//

#ifndef AMBIT_COMPOSITE_TENSOR_H_HPP
#define AMBIT_COMPOSITE_TENSOR_H_HPP

#include "tensor.h"

namespace ambit
{

class CompositeTensor
{
  public:
    CompositeTensor();
    CompositeTensor(const CompositeTensor &other);

    /*
     * Makes a copy of the given tensor. The copy is returned.
     */
    Tensor &add_tensor(Tensor &&new_tensor);

    vector<Tensor>::size_type ntensor() const { return tensors_.size(); }

    bool exists(int idx) const
    {
        if (idx >= 0 && idx < ntensor())
            return true;
        else
            return false;
    }
    Tensor operator()(int idx)
    {
        if (!exists(idx))
        {
            throw std::logic_error("tensor component does not exist");
        }
        return tensors_[idx];
    }

    const Tensor operator()(int idx) const
    {
        if (!exists(idx))
        {
            throw std::logic_error("tensor component does not exist");
        }
        return tensors_[idx];
    }

  protected:
    string name_;
    vector<Tensor> tensors_;
};
}

#endif // AMBIT_COMPOSITE_TENSOR_H_HPP
