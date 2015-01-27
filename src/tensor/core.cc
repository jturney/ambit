#include "tensorimpl.h"
#include "core.h"
#include "memory.h"
#include "math/math.h"

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

    CoreTensorContractionTopology manager(topology,*this,*(const CoreTensorImplPtr)A,*(const CoreTensorImplPtr)B);
    manager.contract(alpha,beta);
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

void CoreTensorContractionTopology::contract(double alpha, double beta)
{
    double* Ap = A_.data();
    double* Bp = B_.data();
    double* Cp = C_.data();
    for (size_t P = 0L; P < ABC_size_; P++) {

        char transL = (A_transpose_ ? 'T' : 'N');
        char transR = (B_transpose_ ? 'T' : 'N');
        size_t nrow = AC_size_;
        size_t ncol = BC_size_;
        double* Lp = Ap;
        double* Rp = Bp;
        size_t ldaL = (A_transpose_ ? AC_size_ : AB_size_);
        size_t ldaR = (B_transpose_ ? AB_size_ : BC_size_);

        size_t nzip = AB_size_;
        size_t ldaC = (C_transpose_ ? AC_size_ : BC_size_);

        if (C_transpose_) {
            std::swap(transL,transR);
            std::swap(nrow,ncol);
            std::swap(Lp,Rp);
            std::swap(ldaL,ldaR);
        }

        C_DGEMM(transL,transR,nrow,ncol,nzip,alpha,Lp,ldaL,Rp,ldaR,beta,Cp,ldaC);

        Cp += AC_size_ * BC_size_;
        Ap += AB_size_ * AC_size_;
        Bp += AB_size_ * BC_size_;
    }
}

}
