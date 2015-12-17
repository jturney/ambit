#include <ambit/tensor.h>

namespace ambit
{

SlicedTensor::SlicedTensor(Tensor T, const IndexRange &range, double factor)
    : T_(T), range_(range), factor_(factor)
{
    if (T_.rank() != range_.size())
    {
        throw std::runtime_error("Sliced tensor does not have correct number "
                                 "of indices for underlying tensor's rank\n "
                                 "range_ " +
                                 std::to_string(range_.size()) + " rank " +
                                 std::to_string(T.rank()));
    }
    for (size_t ind = 0; ind < T.rank(); ind++)
    {
        if (range_[ind].size() != 2L)
            throw std::runtime_error("Each index of an IndexRange should have "
                                     "two elements {start,end+1} in it.");
        if (range_[ind][0] > range_[ind][1])
            throw std::runtime_error(
                "Each index of an IndexRange should end+1>=start in it.");
        if (range_[ind][1] > T_.dims()[ind])
            throw std::runtime_error("IndexRange exceeds size of tensor.");
    }
}

void SlicedTensor::operator=(const SlicedTensor &rhs)
{
    if (T() == rhs.T())
        throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Sliced tensors do not have same rank");
    T_.slice(rhs.T(), range_, rhs.range_, rhs.factor_, 0.0);
}

void SlicedTensor::operator+=(const SlicedTensor &rhs)
{
    if (T() == rhs.T())
        throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Sliced tensors do not have same rank");
    T_.slice(rhs.T(), range_, rhs.range_, rhs.factor_, 1.0);
}

void SlicedTensor::operator-=(const SlicedTensor &rhs)
{
    if (T() == rhs.T())
        throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Sliced tensors do not have same rank");
    T_.slice(rhs.T(), range_, rhs.range_, -rhs.factor_, 1.0);
}
}
