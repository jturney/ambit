#include "cyclops.h"

#define GET_CTF_TENSOR(X) \
    const CyclopsTensorImpl* c##X = dynamic_cast<const CyclopsTensorImpl*>((X)); \
    CTF_Tensor* t##X = c##X->data_;

namespace tensor { namespace cyclops {

namespace globals {
    CTF_World *world = NULL;
    int rank = -1;
    int nprocess = -1;
}

namespace {

    std::string generateGenericLabels(const Dimension& dims)
    {
        std::string labels(dims.size(), 0);
        std::copy(dims.begin(), dims.end(), labels.begin());
        return labels;
    }
    
}

int initialize(int argc, char* argv[])
{
    int flag = 0;
    MPI_Initialized(&flag);

    if (!flag) {
        int error = MPI_Init(&argc, &argv);
        if (error != MPI_SUCCESS) {
            throw std::runtime_error("cyclops::initialize: Unable to initialize MPI.");
        }
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &globals::rank);
    MPI_Comm_size(MPI_COMM_WORLD, &globals::nprocess);

    return 0;
}

CyclopsTensorImpl::CyclopsTensorImpl(const std::string& name,
                                     const Dimension& dims)
    : TensorImpl(Distributed, name, dims)
{
    int *local_sym = new int[dims.size()];
    std::fill(local_sym, local_sym+dims.size(), NS);
    int *local_dims = new int[dims.size()];
    std::copy(dims.begin(), dims.end(), local_dims);

    data_ = new CTF_Tensor(dims.size(),
                           local_dims,
                           local_sym,
                           *globals::world);

    delete[] local_dims;
    delete[] local_sym;
}

CyclopsTensorImpl::~CyclopsTensorImpl()
{
    delete data_;
}

void CyclopsTensorImpl::set_data(double *data, const IndexRange& ranges)
{

}

void CyclopsTensorImpl::get_data(double *data, const IndexRange& ranges) const
{

}

void CyclopsTensorImpl::zero()
{
    *data_ = 0;
}

void CyclopsTensorImpl::scale(double a)
{
    long_int local_size;
    double* local_data;

    local_data = data_->get_raw_data(&local_size);

    for (long_int i=0; i<local_size; ++i) {
        local_data[i] *= a;
    }
}

double CyclopsTensorImpl::norm(double power) const
{
    return 0.0;
}

double CyclopsTensorImpl::rms(double power) const
{
    return 0.0;
}

void CyclopsTensorImpl::scale_and_add(double a, ConstTensorImplPtr x)
{
    typeCheck(Distributed, x);
    dimensionCheck(this, x);

    const CyclopsTensorImpl* cX = dynamic_cast<const CyclopsTensorImpl*>(x);
    CTF_Tensor* tX = cX->data_;

    std::string labels = generateGenericLabels(dims());
    (*data_)[labels.c_str()] += a * (*tX)[labels.c_str()];
}

void CyclopsTensorImpl::pointwise_multiplication(ConstTensorImplPtr x)
{
    typeCheck(Distributed, x);
    dimensionCheck(this, x);

    // ensure this and x are aligned the same
    // since we'll be dealing with raw data

    GET_CTF_TENSOR(x);
    tx->align(*data_);

    long_int local_size, x_size;
    double* local_data, *x_data;

    local_data = data_->get_raw_data(&local_size);
    x_data = tx->get_raw_data(&x_size);

    assert(x_size == local_size);
    for (long_int i=0; i<local_size; ++i) {
        local_data[i] *= x_data[i];
    }
}

void CyclopsTensorImpl::pointwise_division(ConstTensorImplPtr x)
{
    typeCheck(Distributed, x);
    dimensionCheck(this, x);

    // ensure this and x are aligned the same
    // since we'll be dealing with raw data
    // cyclops does not inheritly support
    // pointwise_division ... well any division

    GET_CTF_TENSOR(x);
    tx->align(*data_);

    long_int local_size, x_size;
    double* local_data, *x_data;

    local_data = data_->get_raw_data(&local_size);
    x_data = tx->get_raw_data(&x_size);

    assert(x_size == local_size);
    for (long_int i=0; i<local_size; ++i) {
        local_data[i] /= x_data[i];
    }
}

double CyclopsTensorImpl::dot(ConstTensorImplPtr x) const
{
    typeCheck(Distributed, x);
    dimensionCheck(this, x);

    // ensure this and x are aligned the same
    // since we'll be dealing with raw data

    GET_CTF_TENSOR(x);
    tx->align(*data_);

    long_int local_size, x_size;
    double* local_data, *x_data;
    double local_total = 0, global_total = 0;

    local_data = data_->get_raw_data(&local_size);
    x_data = tx->get_raw_data(&x_size);

    assert(x_size == local_size);
    for (long_int i=0; i<local_size; ++i) {
        local_total += local_data[i] * x_data[i];
    }

    MPI_Allreduce(&local_total, &global_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_total;
}

void CyclopsTensorImpl::contract(ConstTensorImplPtr A, ConstTensorImplPtr B, const ContractionTopology &topology, double alpha, double beta)
{
    // Need a way to translate topology to strings of indices
}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::syev(EigenvalueOrder order) const
{

}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::geev(EigenvalueOrder order) const
{
    
}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::svd() const
{
    
}

TensorImplPtr CyclopsTensorImpl::cholesky() const
{

}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::lu() const
{
}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::qr() const
{

}

TensorImplPtr CyclopsTensorImpl::cholesky_inverse() const
{

}

TensorImplPtr CyclopsTensorImpl::inverse() const
{

}

TensorImplPtr CyclopsTensorImpl::power(double power, double condition) const
{

}

void CyclopsTensorImpl::givens(int dim, int i, int j, double s, double c)
{

}

}}