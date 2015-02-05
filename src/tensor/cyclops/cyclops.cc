#include "cyclops.h"
#include "../macros.h"
#include <El.hpp>

#define GET_CTF_TENSOR(X) \
    const CyclopsTensorImpl* c##X = dynamic_cast<const CyclopsTensorImpl*>((X)); \
    CTF_Tensor* t##X = c##X->data_;

namespace tensor { namespace cyclops {

namespace globals {
    CTF_World *world = NULL;
    int rank = -1;
    int nprocess = -1;

    // did we initialize MPI or did the user?
    int initialized_mpi = 0;
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
    MPI_Initialized(&globals::initialized_mpi);

    if (!globals::initialized_mpi) {
        int error = MPI_Init(&argc, &argv);
        if (error != MPI_SUCCESS) {
            throw std::runtime_error("cyclops::initialize: Unable to initialize MPI.");
        }
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &globals::rank);
    MPI_Comm_size(MPI_COMM_WORLD, &globals::nprocess);

    globals::world = new CTF_World(argc, argv);

    return 0;
}

void finalize()
{
    delete globals::world;
    globals::world = NULL;

    // if we initialized MPI then we finalize it.
    if (!globals::initialized_mpi)
        MPI_Finalize();
}

CyclopsTensorImpl::CyclopsTensorImpl(const std::string& name,
                                     const Dimension& dims)
    : TensorImpl(kDistributed, name, dims)
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

void CyclopsTensorImpl::scale(const double& a)
{
    long_int local_size;
    double* local_data;

    local_data = data_->get_raw_data(&local_size);

    VECTORIZED_LOOP
    for (long_int i=0; i<local_size; ++i) {
        local_data[i] *= a;
    }
}

double CyclopsTensorImpl::norm(double /*power*/) const
{
    // not sure what power is for
    return data_->norm1();
}

double CyclopsTensorImpl::rms(double /*power*/) const
{
    // not sure what power is for
    return data_->norm2();
}

void CyclopsTensorImpl::scale_and_add(const double& a, ConstTensorImplPtr x)
{
    typeCheck(kDistributed, x);
    dimensionCheck(this, x);

    const CyclopsTensorImpl* cX = dynamic_cast<const CyclopsTensorImpl*>(x);
    CTF_Tensor* tX = cX->data_;

    std::string labels = generateGenericLabels(dims());
    (*data_)[labels.c_str()] += a * (*tX)[labels.c_str()];
}

void CyclopsTensorImpl::permute(
        ConstTensorImplPtr A,
        const std::vector<std::string>& Cinds,
        const std::vector<std::string>& Ainds,
        double alpha,
        double beta)
{

}

void CyclopsTensorImpl::slice(
        ConstTensorImplPtr A,
        const IndexRange& Cinds,
        const IndexRange& Ainds,
        double alpha,
        double beta)
{

}

void CyclopsTensorImpl::pointwise_multiplication(ConstTensorImplPtr x)
{
    typeCheck(kDistributed, x);
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
    typeCheck(kDistributed, x);
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
    typeCheck(kDistributed, x);
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

void CyclopsTensorImpl::contract(
        ConstTensorImplPtr A,
        ConstTensorImplPtr B,
        const std::vector<std::string>& Cinds,
        const std::vector<std::string>& Ainds,
        const std::vector<std::string>& Binds,
        double alpha,
        double beta)
{

}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::syev(EigenvalueOrder order) const
{
    TensorImpl::squareCheck(this);
    size_t length = dims()[0];

#if defined(HAVE_ELEMENTAL)
    // since elemental and cyclops distribute their data
    // differently. construct an elemental matrix and
    // use its data layout to pull remote data from
    // cyclops.

    El::DistMatrix<double> H(length, length);
    copyToElemental2(H);

    // construct elemental storage for values and vectors
    El::DistMatrix<double,El::VR,El::STAR> w;
    El::DistMatrix<double> X;
    El::SortType sort = order == kAscending ? El::ASCENDING : El::DESCENDING;
    El::HermitianEig(El::LOWER, H, w, X, sort);

    El::Print(H, "H");
    El::Print(X, "X");
    El::Print(w, "w");

    // construct cyclops tensors to hold eigen vectors and values.
    Dimension value_dims(1);
    value_dims[0] = length;
    CyclopsTensorImpl* vectors = new CyclopsTensorImpl("Eigenvectors", dims());
    CyclopsTensorImpl* values = new CyclopsTensorImpl("Eigenvalues", value_dims);

    // populate cyclops with elemental data
    vectors->copyFromElemental2(X);
    values->copyFromElemental1(w);

    std::map<std::string, TensorImplPtr> results;
    results["values"] = values;
    results["vectors"] = vectors;

    return results;
#else

    // Distributed LAPACK is not available.
    // Copy all data to master node and use
    // local LAPACK then broadcast the data
    // back out to the nodes.
    if (globals::rank() == 0) {
        int info;

    }

#endif
}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::geev(EigenvalueOrder order) const
{
    TensorImpl::squareCheck(this);
}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::svd() const
{
    TensorImpl::rankCheck(2, this);
    size_t rows = dims()[0];
    size_t cols = dims()[1];

#if defined(HAVE_ELEMENTAL)
    // since elemental and cyclops distribute their data
    // differently. construct an elemental matrix and
    // use its data layout to pull remote data from
    // cyclops.

    std::vector<kv_pair> pairs;
    El::DistMatrix<double> U(rows, cols);
    El::DistMatrix<double> V;
    El::DistMatrix<double, El::VR, El::STAR> s;

    copyToElemental2(U);

    El::SVD(U, s, V);

    Dimension sdim(1);
    sdim[0] = dims()[0];
    Dimension Vtdim(2);
    Vtdim[0] = dims()[1];
    Vtdim[1] = dims()[0];

    CyclopsTensorImpl* tU = new CyclopsTensorImpl("U", dims());
    CyclopsTensorImpl* ts = new CyclopsTensorImpl("s", sdim);
    CyclopsTensorImpl* tVt = new CyclopsTensorImpl("Vt", Vtdim);

    tU->copyFromElemental2(U);
    ts->copyFromElemental1(s);
    tVt->copyFromElemental2(V);

    std::map<std::string, TensorImplPtr> results;
    results["U"] = tU;
    results["s"] = ts;
    results["Vt"] = tVt;

    return results;
#endif
}

TensorImplPtr CyclopsTensorImpl::cholesky() const
{

}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::lu() const
{
    TensorImpl::rankCheck(2, this);
    size_t rows = dims()[0];
    size_t cols = dims()[1];

#if defined(HAVE_ELEMENTAL)
    // since elemental and cyclops distribute their data
    // differently. construct an elemental matrix and
    // use its data layout to pull remote data from
    // cyclops.

    std::vector<kv_pair> pairs;
    El::DistMatrix<double> A(rows, cols);
    copyToElemental2(A);

    El::LU(A);

    CyclopsTensorImpl* result = new CyclopsTensorImpl("A", dims());
    result->copyFromElemental2(A);

    std::map<std::string, TensorImplPtr> results;
    results["A"] = result;
    return results;
#endif
}

std::map<std::string, TensorImplPtr> CyclopsTensorImpl::qr() const
{

}

TensorImplPtr CyclopsTensorImpl::cholesky_inverse() const
{

}

TensorImplPtr CyclopsTensorImpl::inverse() const
{
    TensorImpl::rankCheck(2, this);
    size_t rows = dims()[0];
    size_t cols = dims()[1];

#if defined(HAVE_ELEMENTAL)
    // since elemental and cyclops distribute their data
    // differently. construct an elemental matrix and
    // use its data layout to pull remote data from
    // cyclops.

    std::vector<kv_pair> pairs;
    El::DistMatrix<double> A(rows, cols);
    copyToElemental2(A);

    El::Inverse(A);

    CyclopsTensorImpl* result = new CyclopsTensorImpl("A", dims());
    result->copyFromElemental2(A);

    return result;
#endif
}

TensorImplPtr CyclopsTensorImpl::power(double power, double condition) const
{

}

void CyclopsTensorImpl::givens(int dim, int i, int j, double s, double c)
{

}

#if defined(HAVE_ELEMENTAL)
void CyclopsTensorImpl::copyToElemental2(El::DistMatrix<double> &U) const
{
    rankCheck(2, this);
    size_t cols = dims()[1];

    std::vector<kv_pair> pairs;
    const int cshift = U.ColShift();
    const int rshift = U.RowShift();
    const int cstride = U.ColStride();
    const int rstride = U.RowStride();

    // determine which pairs from cyclops we need.
    for (int i=0; i<U.LocalHeight(); i++) {
        for (int j=0; j<U.LocalWidth(); j++) {
            const int c = cshift+i*cstride;
            const int r = rshift+j*rstride;

            pairs.push_back(kv_pair(r*cols+c, 0));
        }
    }

    // gather data from cyclops
    data_->read(pairs.size(), pairs.data());

    // populate elemental
    for (size_t p=0; p<pairs.size(); p++) {
        const int r = pairs[p].k/cols;
        const int c = pairs[p].k-cols*r;
        const int i = (c-cshift)/cstride;
        const int j = (r-rshift)/rstride;

        U.SetLocal(i, j, pairs[p].d);
    }
}

void CyclopsTensorImpl::copyFromElemental2(const El::DistMatrix<double> &X)
{
    size_t length = dims()[1];
    std::vector<kv_pair> pairs;
    const int cshift = X.ColShift();
    const int rshift = X.RowShift();
    const int cstride = X.ColStride();
    const int rstride = X.RowStride();

    for (int i=0; i<X.LocalHeight(); i++) {
        for (int j=0; j<X.LocalWidth(); j++) {
            const int c = cshift+i*cstride;
            const int r = rshift+j*rstride;

            pairs.push_back(kv_pair(r*length+c, X.GetLocal(i, j)));
        }
    }
    data_->write(pairs.size(), pairs.data());
}

void CyclopsTensorImpl::copyFromElemental1(const El::DistMatrix<double, El::VR, El::STAR>& w)
{
    std::vector<kv_pair> pairs;
    const int cshift = w.ColShift();
    for (int i=0; i<w.LocalHeight(); i++) {
        pairs.push_back(kv_pair(i+cshift, w.GetLocal(i, 0)));
    }
    data_->write(pairs.size(), pairs.data());
}
#endif

}}
