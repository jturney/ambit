#include "tensorimpl.h"
#include "core.h"
#include "memory.h"

namespace tensor {

CoreTensorImpl::CoreTensorImpl(const std::string& name, const Dimension& dims)
        : TensorImpl(kCore, name, dims)
{
    data_ = memory::allocate<double>(numel());
    memset(data_,'\0', sizeof(double)*numel());
}
CoreTensorImpl::~CoreTensorImpl()
{
    if (data_) memory::free(data_);
}
void CoreTensorImpl::set_data(double* data, const IndexRange& range)
{
    if (range.size() == 0) {
        memcpy(data_,data,sizeof(double)*numel());
        return;
    }
    // TODO
}
void CoreTensorImpl::get_data(double* data, const IndexRange& range) const
{
    if (range.size() == 0) {
        memcpy(data,data_,sizeof(double)*numel());
        return;
    }
    // TODO
}
void CoreTensorImpl::zero()
{
    memset(data_,'\0', sizeof(double)*numel());
}
void CoreTensorImpl::scale(const double& a)
{
    VECTORIZED_LOOP
    for (size_t i=0, end=numel(); i<end; ++i) {
        data_[i] *= a;
    }
}
double CoreTensorImpl::norm(double power) const
{
    ThrowNotImplementedException;
}
double CoreTensorImpl::rms(double power) const
{
    ThrowNotImplementedException;
}
void CoreTensorImpl::scale_and_add(double a, ConstTensorImplPtr x)
{
    ThrowNotImplementedException;
}
void CoreTensorImpl::pointwise_multiplication(ConstTensorImplPtr x)
{
    ThrowNotImplementedException;
}
void CoreTensorImpl::pointwise_division(ConstTensorImplPtr x)
{
    ThrowNotImplementedException;
}
double CoreTensorImpl::dot(ConstTensorImplPtr x) const
{
    ThrowNotImplementedException;
}
void CoreTensorImpl::contract(ConstTensorImplPtr A, ConstTensorImplPtr B, const ContractionTopology &topology,
                              double alpha, double beta)
{
    ThrowNotImplementedException;
}

std::map<std::string, TensorImplPtr> CoreTensorImpl::syev(EigenvalueOrder order) const
{
    ThrowNotImplementedException;
}
std::map<std::string, TensorImplPtr> CoreTensorImpl::geev(EigenvalueOrder order) const
{
    ThrowNotImplementedException;
}
std::map<std::string, TensorImplPtr> CoreTensorImpl::svd() const
{
    ThrowNotImplementedException;
}

TensorImplPtr CoreTensorImpl::cholesky() const
{
    ThrowNotImplementedException;
}

std::map<std::string, TensorImplPtr> CoreTensorImpl::lu() const
{
    ThrowNotImplementedException;
}
std::map<std::string, TensorImplPtr> CoreTensorImpl::qr() const
{
    ThrowNotImplementedException;
}

TensorImplPtr CoreTensorImpl::cholesky_inverse() const
{
    ThrowNotImplementedException;
}

TensorImplPtr CoreTensorImpl::inverse() const
{
    ThrowNotImplementedException;
}
TensorImplPtr CoreTensorImpl::power(double power, double condition) const
{
    ThrowNotImplementedException;
}

void CoreTensorImpl::givens(int dim, int i, int j, double s, double c)
{
    ThrowNotImplementedException;
}

}
