#include <tensor/tensor.h>
#include "tensorimpl.h"
#include "core.h"
#include "disk.h"
#include "indices.h"

// include header files to specific tensor types supported.
#if defined(HAVE_CYCLOPS)
#   include "cyclops/cyclops.h"
#endif

#include <cstdlib>
#include <list>
#include <algorithm>

namespace tensor {

namespace settings {

int nprocess = 1;

int rank = 0;

bool debug = false;

size_t memory = 1 * 1024 * 1024 * 1024;

}

int initialize(int argc, char** argv)
{
    /// Set the scratch path for disk files
    const char* scratch_env = std::getenv("TENSOR_SCRATCH");
    if (scratch_env != nullptr) {
        std::string scratch_str(scratch_env);
        Tensor::set_scratch_path(scratch_str);
    } else {
        Tensor::set_scratch_path(".");
    }

#if defined(HAVE_CYCLOPS)
    cyclops::initialize(argc, argv);
#endif

    return 0;
}

void finalize()
{
#if defined(HAVE_CYCLOPS)
    cyclops::finalize();
#endif
}

std::string Tensor::scratch_path__ = ".";

Tensor::Tensor(shared_ptr<TensorImpl> tensor)
    : tensor_(tensor)
{}

Tensor Tensor::build(TensorType type, const std::string& name, const Dimension& dims)
{
    Tensor newObject;

    if (type == kAgnostic) {
        #if defined(HAVE_CYCLOPS)
        type = kDistributed;
        #else
        type = kCore;
        #endif
    }
    switch(type) {
        case kCore:
//            printf("Constructing core tensor.\n");
            newObject.tensor_.reset(new CoreTensorImpl(name, dims));
            break;

        case kDisk:
//            printf("Constructing disk tensor.\n");
            newObject.tensor_.reset(new DiskTensorImpl(name, dims));
            break;

        case kDistributed:
//            printf("Constructing distributed tensor.\n");

            #if defined(HAVE_CYCLOPS)
            newObject.tensor_.reset(new cyclops::CyclopsTensorImpl(name, dims));
            #else
            throw std::runtime_error("Tensor::build: Unable to construct distributed tensor object");
            #endif

            break;

        default:
            throw std::runtime_error("Tensor::build: Unknown parameter passed into 'type'.");
    }

    return newObject;
}

Tensor Tensor::clone(TensorType type) const
{
    Tensor current = Tensor::build(type, name(), dims());
    current.copy(*this);
    return current;
}

void Tensor::copy(const Tensor& other)
{
    tensor_->copy(other.tensor_.get());
}

Tensor::Tensor()
{}

TensorType Tensor::type() const
{
    return tensor_->type();
}

std::string Tensor::name() const
{
    return tensor_->name();
}

void Tensor::set_name(const std::string& name)
{
    tensor_->set_name(name);
}

const std::vector<size_t>& Tensor::dims() const
{
    return tensor_->dims();
}

size_t Tensor::dim(size_t ind) const
{
    return tensor_->dim(ind);
}

size_t Tensor::rank() const
{
    return tensor_->dims().size();
}

size_t Tensor::numel() const
{
    return tensor_->numel();
}

void Tensor::print(FILE *fh, bool level, std::string const &format, int maxcols) const
{
    tensor_->print(fh, level, format, maxcols);
}

LabeledTensor Tensor::operator()(const std::string& indices)
{
    return LabeledTensor(*this, indices::split(indices));
}

SlicedTensor Tensor::operator()(const IndexRange& range)
{
    return SlicedTensor(*this, range);
}

SlicedTensor Tensor::operator()()
{
    IndexRange range;
    for (size_t ind = 0L; ind < rank(); ind++) {
        range.push_back({0L,dim(ind)});
    }
    return SlicedTensor(*this, range);
}

std::vector<double>& Tensor::data()
{
    return tensor_->data();
}

const std::vector<double>& Tensor::data() const
{
    return tensor_->data();
}

Tensor Tensor::cat(std::vector<Tensor> const, int dim)
{
    ThrowNotImplementedException;
}

double Tensor::norm(int type) const
{
    return tensor_->norm(type);
}
void Tensor::zero()
{
    tensor_->scale(0.0);
}

void Tensor::scale(double a)
{
    tensor_->scale(a);
}

std::map<std::string, Tensor> Tensor::map_to_tensor(const std::map<std::string, TensorImplPtr>& x)
{
    std::map<std::string, Tensor> result;

    for (std::map<std::string, TensorImplPtr>::const_iterator iter = x.begin();
            iter != x.end();
            ++iter) {
        result.insert(make_pair(iter->first, Tensor(shared_ptr<TensorImpl>(iter->second))));
    }
    return result;
}

std::map<std::string, Tensor> Tensor::syev(EigenvalueOrder order) const
{
    return map_to_tensor(tensor_->syev(order));
}

//std::map<std::string, Tensor> Tensor::geev(EigenvalueOrder order) const
//{
//    return map_to_tensor(tensor_->geev(order));
//}
//
//std::map<std::string, Tensor> Tensor::svd() const
//{
//    return map_to_tensor(tensor_->svd());
//}
//
//Tensor Tensor::cholesky() const
//{
//    return Tensor(shared_ptr<TensorImpl>(tensor_->cholesky()));
//}
//
//std::map<std::string, Tensor> Tensor::lu() const
//{
//    return map_to_tensor(tensor_->lu());
//}
//
//std::map<std::string, Tensor> Tensor::qr() const
//{
//    return map_to_tensor(tensor_->qr());
//}
//
//Tensor Tensor::cholesky_inverse() const
//{
//    return Tensor(shared_ptr<TensorImpl>(tensor_->cholesky_inverse()));
//}
//
//Tensor Tensor::inverse() const
//{
//    return Tensor(shared_ptr<TensorImpl>(tensor_->inverse()));
//}

Tensor Tensor::power(double alpha, double condition) const
{
    return Tensor(shared_ptr<TensorImpl>(tensor_->power(alpha, condition)));
}

void Tensor::contract(
    const Tensor& A,
    const Tensor& B,
    const std::vector<std::string>& Cinds,
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Binds,
    double alpha,
    double beta)
{
    tensor_->contract(
        A.tensor_.get(),
        B.tensor_.get(),
        Cinds,
        Ainds,
        Binds,
        alpha,
        beta);
}
void Tensor::permute(
    const Tensor &A,
    const std::vector<std::string>& Cinds,
    const std::vector<std::string>& Ainds,
    double alpha,
    double beta)
{
    tensor_->permute(A.tensor_.get(),Cinds,Ainds,alpha,beta);
}
void Tensor::slice(
    const Tensor &A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha,
    double beta)
{
    tensor_->slice(A.tensor_.get(),Cinds,Ainds,alpha,beta);
}
void Tensor::gemm(
    const Tensor& A,
    const Tensor& B,
    bool transA,
    bool transB,
    size_t nrow,
    size_t ncol,
    size_t nzip,
    size_t ldaA,
    size_t ldaB,
    size_t ldaC,
    size_t offA,
    size_t offB,
    size_t offC,
    double alpha,
    double beta)
{
    tensor_->gemm(
        A.tensor_.get(),
        B.tensor_.get(),
        transA,
        transB,
        nrow,
        ncol,
        nzip,
        ldaA,
        ldaB,
        ldaC,
        offA,
        offB,
        offC,
        alpha,
        beta);
}

bool Tensor::operator==(const Tensor& other) const
{
    return tensor_ == other.tensor_;
}

bool Tensor::operator!=(const Tensor& other) const
{
    return tensor_ != other.tensor_;
}

}
