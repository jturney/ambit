#include <tensor/tensor.h>
#include "tensorimpl.h"
#include "core.h"
#include "indices.h"

// include header files to specific tensor types supported.
#if defined(HAVE_CYCLOPS)
#   include "cyclops/cyclops.h"
#endif

#include <list>
#include <algorithm>

namespace tensor {

LabeledTensor::LabeledTensor(Tensor& T, const std::vector<std::string>& indices, double factor) :
        T_(T), indices_(indices), factor_(factor)
{
    if (T_.rank() != indices.size()) throw std::runtime_error("Labeled tensor does not have correct number of indices for underlying tensor's rank");
}

void LabeledTensor::operator=(const LabeledTensor& rhs)
{
    if (T() == rhs.T()) throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank()) throw std::runtime_error("Permuted tensors do not have same rank");
    T_.permute(rhs.T(),indices_,rhs.indices_);
    T_.scale(rhs.factor());
}

void LabeledTensor::operator+=(const LabeledTensor& rhs)
{
    if (indices::equivalent(indices_, rhs.indices_) == true) {
        T_.scale_and_add(rhs.factor(), rhs.T());
    }
    else {
        // TODO: Sort and scaling.
        ThrowNotImplementedException;
    }
}

void LabeledTensor::operator-=(const LabeledTensor& rhs)
{
    if (indices::equivalent(indices_, rhs.indices_) == true) {
        T_.scale_and_add(-rhs.factor(), rhs.T());
    }
    else {
        // TODO: Sort and scaling.
        ThrowNotImplementedException;
    }
}

LabeledTensorProduct LabeledTensor::operator*(const LabeledTensor &rhs)
{
    return LabeledTensorProduct(*this, rhs);
}

LabeledTensorAddition LabeledTensor::operator+(const LabeledTensor &rhs)
{
    return LabeledTensorAddition(*this, rhs);
}

LabeledTensorSubtraction LabeledTensor::operator-(const LabeledTensor &rhs)
{
    return LabeledTensorSubtraction(*this, rhs);
}

void LabeledTensor::operator=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    // 1. create a ContractionTopology
    ContractionTopology ct(*this, A, B);

    T_.contract(A.T(),
                B.T(),
                indices(),
                A.indices(),
                B.indices(),
                A.factor() * B.factor(),
                0.0);
    
    // 2. call contract on the tensor.
    T_.contract(A.T(),
                B.T(),
                ct,
                A.factor() * B.factor(),
                0.0);
                 
}

void LabeledTensor::operator+=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    // 1. create a ContractionTopology
    ContractionTopology ct(*this, A, B);

    T_.contract(A.T(),
                B.T(),
                indices(),
                A.indices(),
                B.indices(),
                A.factor() * B.factor(),
                1.0);
    
    // 2. call contract on the tensor.
    T_.contract(A.T(),
                B.T(),
                ct,
                A.factor() * B.factor(),
                1.0);
}

void LabeledTensor::operator-=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    // 1. create a ContractionTopology
    ContractionTopology ct(*this, A, B);

    T_.contract(A.T(),
                B.T(),
                indices(),
                A.indices(),
                B.indices(),
                - A.factor() * B.factor(),
                1.0);

    // 2. call contract on the tensor.
    T_.contract(A.T(),
                B.T(),
                ct,
                - A.factor() * B.factor(),
                1.0);

}

void LabeledTensor::operator=(const LabeledTensorAddition& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator+=(const LabeledTensorAddition& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator-=(const LabeledTensorAddition& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator=(const LabeledTensorSubtraction& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator+=(const LabeledTensorSubtraction& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator-=(const LabeledTensorSubtraction& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator*=(const double& scale)
{
    T_.scale(scale);
}

void LabeledTensor::operator/=(const double& scale)
{
    T_.scale(1.0/scale);
}

}
