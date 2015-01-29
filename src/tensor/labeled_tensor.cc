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

void LabeledTensor::operator=(const LabeledTensor& rhs)
{
    if (indices::equivalent(indices_, rhs.indices_) == true) {
        // equivalent indices:   "i,a" = "i,a"
        // perform a simple copy
        T_.copy(rhs.T(), rhs.factor());
    }
    else {
        // TODO: potential sorting of data
        printf("Potential sorting assignment.\n");
        ThrowNotImplementedException;
    }
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
