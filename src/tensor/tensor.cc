#include <cstdlib>
#include <list>
#include <algorithm>

#include <ambit/tensor.h>
#include <ambit/print.h>
#include "tensorimpl.h"
#include "core/core.h"
#include "disk/disk.h"
#include "indices.h"

#include "globals.h"

#include <ambit/timer.h>

// include header files to specific tensor types supported.
#if defined(HAVE_CYCLOPS)
#   include "cyclops/cyclops.h"
#endif

namespace ambit {

namespace settings {

namespace {

int ninitialized = 0;

}

int nprocess = 1;

int rank = 0;

bool debug = false;

size_t memory = 1 * 1024 * 1024 * 1024;

#if defined(HAVE_CYCLOPS)
const bool distributed_capable = true;
#else
const bool distributed_capable = false;
#endif

bool timers = false;

}

namespace {

void common_initialize(int /*argc*/, char* const * /*argv*/)
{
    if (settings::ninitialized != 0)
        throw std::runtime_error("ambit::initialize: Ambit has already been initialized.");

    settings::ninitialized++;

    timer::initialize();

    // Set the scratch path for disk files
    const char* scratch_env = std::getenv("TENSOR_SCRATCH");
    if (scratch_env != nullptr) {
        std::string scratch_str(scratch_env);
        Tensor::set_scratch_path(scratch_str);
    } else {
        Tensor::set_scratch_path(".");
    }
}

}

int initialize(int argc, char* * argv)
{
    common_initialize(argc, argv);

#if defined(HAVE_CYCLOPS)
    return cyclops::initialize(argc, argv);
#else
    return 0;
#endif
}

void finalize()
{
    if (settings::ninitialized == 0)
        throw std::runtime_error("ambit::finalize: Ambit has already been finalized.");

    settings::ninitialized--;

#if defined(HAVE_CYCLOPS)
    cyclops::finalize();
#endif

    timer::report();
    timer::finalize();
}

void barrier()
{
#if defined(HAVE_MPI)
    MPI_Barrier(globals::communicator);
#endif
}

std::string Tensor::scratch_path__ = ".";

Tensor::Tensor(shared_ptr<TensorImpl> tensor)
    : tensor_(tensor)
{}

Tensor Tensor::build(TensorType type, const std::string& name, const Dimension& dims)
{
    if (settings::ninitialized == 0)
        throw std::runtime_error("ambit::Tensor::build: Ambit has not been initialized.");

    ambit::timer::timer_push("Tensor::build");

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
            newObject.tensor_.reset(new CoreTensorImpl(name, dims));
            break;

        case kDisk:
            newObject.tensor_.reset(new DiskTensorImpl(name, dims));
            break;

        case kDistributed:
            #if defined(HAVE_CYCLOPS)
            newObject.tensor_.reset(new cyclops::CyclopsTensorImpl(name, dims));
            #else
            throw std::runtime_error("Tensor::build: Unable to construct distributed tensor object");
            #endif

            break;

        default:
            throw std::runtime_error("Tensor::build: Unknown parameter passed into 'type'.");
    }

    ambit::timer::timer_pop();

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

LabeledTensor Tensor::operator()(const std::string& indices) const
{
    return LabeledTensor(*this, indices::split(indices));
}

SlicedTensor Tensor::operator()(const IndexRange& range) const
{
    return SlicedTensor(*this, range);
}

SlicedTensor Tensor::operator()() const
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
    timer::timer_push("Tensor::norm");
    auto result = tensor_->norm(type);
    timer::timer_pop();
    return result;
}
void Tensor::zero()
{
    timer::timer_push("Tensor::zero");
    tensor_->scale(0.0);
    timer::timer_pop();
}

void Tensor::scale(double a)
{
    timer::timer_push("Tensor::scale");
    tensor_->scale(a);
    timer::timer_pop();
}

void Tensor::set(double alpha)
{
    timer::timer_push("Timer::set");
    tensor_->set(alpha);
    timer::timer_pop();
}

void Tensor::iterate(const std::function<void (const std::vector<size_t>&, double&)>& func)
{
    timer::timer_push("Tensor::iterate");
    tensor_->iterate(func);
    timer::timer_pop();
}

void Tensor::citerate(const std::function<void (const std::vector<size_t>&, const double&)>& func) const
{
    timer::timer_push("Tensor::citerate");
    tensor_->citerate(func);
    timer::timer_pop();
}

std::tuple<double, std::vector<size_t>> Tensor::max() const
{
    timer::timer_push("Tensor::max");
    auto result = tensor_->max();
    timer::timer_pop();

    return result;
}

std::tuple<double, std::vector<size_t>> Tensor::min() const
{
    timer::timer_push("Tensor::min");
    auto result = tensor_->min();
    timer::timer_pop();

    return result;
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
    timer::timer_push("Tensor::syev");
    auto result = map_to_tensor(tensor_->syev(order));
    timer::timer_pop();
    return result;
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
    const Indices& Cinds,
    const Indices& Ainds,
    const Indices& Binds,
    double alpha,
    double beta)
{
    if (ambit::settings::debug)
        ambit::print("    #: " + std::to_string(beta) + " " + name() + "[" + indices::to_string(Cinds) + "] = " + std::to_string(alpha) + " " + A.name() + "[" + indices::to_string(Ainds) + "] * " + B.name() + "[" + indices::to_string(Binds) + "]\n");

    timer::timer_push("#: " + std::to_string(beta) + " " + name() + "[" + indices::to_string(Cinds) + "] = " + std::to_string(alpha) + " " + A.name() + "[" + indices::to_string(Ainds) + "] * " + B.name() + "[" + indices::to_string(Binds) + "]");

    tensor_->contract(
        A.tensor_.get(),
        B.tensor_.get(),
        Cinds,
        Ainds,
        Binds,
        alpha,
        beta);

    timer::timer_pop();
}
void Tensor::permute(
    const Tensor &A,
    const Indices& Cinds,
    const Indices& Ainds,
    double alpha,
    double beta)
{
    if (ambit::settings::debug)
        ambit::print("    P: " + name() + "[" + indices::to_string(Cinds) + "] = " + A.name() + "[" + indices::to_string(Ainds) + "]\n");
    timer::timer_push("P: " + name() + "[" + indices::to_string(Cinds) + "] = " + A.name() + "[" + indices::to_string(Ainds) + "]");

    tensor_->permute(A.tensor_.get(),Cinds,Ainds,alpha,beta);

    timer::timer_pop();
}
void Tensor::slice(
    const Tensor &A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha,
    double beta)
{
    timer::timer_push("Tensor::slice");

    tensor_->slice(A.tensor_.get(),Cinds,Ainds,alpha,beta);

    timer::timer_pop();
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
    timer::timer_push("Tensor::gemm");
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

    timer::timer_pop();
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
