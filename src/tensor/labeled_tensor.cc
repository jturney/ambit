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

LabeledTensorAddition LabeledTensor::operator-(const LabeledTensor &rhs)
{
    return LabeledTensorAddition(*this, -rhs);
}

void LabeledTensor::operator=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    T_.contract(A.T(),
                B.T(),
                indices(),
                A.indices(),
                B.indices(),
                A.factor() * B.factor(),
                0.0);
    
}

void LabeledTensor::operator+=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    T_.contract(A.T(),
                B.T(),
                indices(),
                A.indices(),
                B.indices(),
                A.factor() * B.factor(),
                1.0);
    
}

void LabeledTensor::operator-=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    T_.contract(A.T(),
                B.T(),
                indices(),
                A.indices(),
                B.indices(),
                - A.factor() * B.factor(),
                1.0);

}

void LabeledTensor::operator=(const LabeledTensorAddition& rhs)
{
    T().zero();
    for (size_t ind=0, end=rhs.size(); ind < end; ++ind) {
        const LabeledTensor& labeledTensor = rhs[ind];

        if (T() == labeledTensor.T()) throw std::runtime_error("Self assignment is not allowed.");
        if (indices::equivalent(indices_, labeledTensor.indices()) == false) {
            throw std::runtime_error("Indices must be equivalent.");
        }

        T_.scale_and_add(labeledTensor.factor(), labeledTensor.T());
    }
}

void LabeledTensor::operator+=(const LabeledTensorAddition& rhs)
{
    for (size_t ind=0, end=rhs.size(); ind < end; ++ind) {
        const LabeledTensor& labeledTensor = rhs[ind];

        if (T() == labeledTensor.T()) throw std::runtime_error("Self assignment is not allowed.");
        if (indices::equivalent(indices_, labeledTensor.indices()) == false) {
            throw std::runtime_error("Indices must be equivalent.");
        }

        T_.scale_and_add(labeledTensor.factor(), labeledTensor.T());
    }
}

void LabeledTensor::operator-=(const LabeledTensorAddition& rhs)
{
    for (size_t ind=0, end=rhs.size(); ind < end; ++ind) {
        const LabeledTensor& labeledTensor = rhs[ind];

        if (T() == labeledTensor.T()) throw std::runtime_error("Self assignment is not allowed.");
        if (indices::equivalent(indices_, labeledTensor.indices()) == false) {
            throw std::runtime_error("Indices must be equivalent.");
        }

        T_.scale_and_add(-labeledTensor.factor(), labeledTensor.T());
    }
}

void LabeledTensor::operator*=(const double& scale)
{
    T_.scale(scale);
}

void LabeledTensor::operator/=(const double& scale)
{
    T_.scale(1.0/scale);
}

LabeledTensorDistributive LabeledTensor::operator*(const LabeledTensorAddition& rhs)
{
    return LabeledTensorDistributive(*this, rhs);
}

void LabeledTensor::operator=(const LabeledTensorDistributive &rhs)
{
    T_.zero();

    for (const LabeledTensor& B : rhs.B()) {
        *this += const_cast<LabeledTensor&>(rhs.A()) * const_cast<LabeledTensor&>(B);
    }
}

LabeledTensorDistributive LabeledTensorAddition::operator*(const LabeledTensor& other)
{
    return LabeledTensorDistributive(other, *this);
}

LabeledTensorAddition& LabeledTensorAddition::operator*(const double& scalar)
{
    // distribute the scalar to each term
    for (LabeledTensor& T : tensors_) {
        T *= scalar;
    }

    return *this;
}

LabeledTensorAddition& LabeledTensorAddition::operator-()
{
    for (LabeledTensor& T : tensors_) {
        T *= -1.0;
    }

    return *this;
}

LabeledTensorProduct::operator double() const
{
    // Only handles binary expressions.
    if (size() == 0 || size() > 2) throw std::runtime_error("Conversion operator only supports binary expressions at the moment.");
    if (indices::equivalent(tensors_[0].indices(), tensors_[1].indices()) == false) {
        throw std::runtime_error("Conversion operator implies dot product and thus equivalent indices.");
    }

    return tensors_[0].T().dot(tensors_[1].T());
}

}
