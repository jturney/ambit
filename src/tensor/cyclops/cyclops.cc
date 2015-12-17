#if !defined(HAVE_CYCLOPS)
#error The Cyclops interface is being compiled without Cyclops present.
#endif

#include "cyclops.h"
#include "../globals.h"
#include <ambit/print.h>

#define GET_CTF_TENSOR(X)                                                      \
    const CyclopsTensorImpl *c##X =                                            \
        dynamic_cast<const CyclopsTensorImpl *>((X));                          \
    CTF_Tensor *t##X = c##X->cyclops_;

namespace ambit
{
namespace cyclops
{

namespace details
{

CTF_World *world = nullptr;

// did we initialize MPI or did the user?
int initialized_mpi = 0;
}

namespace
{

std::vector<std::string> generateCyclopsLabels(const std::vector<Indices> &inds)
{
    static const char *cyclops_index_set =
        "abcdefghijklmnopqrstuvwyxzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    int index = 0;
    std::map<std::string, std::string> index_map;
    for (size_t k = 0L; k < inds.size(); k++)
    {
        const std::vector<std::string> &ind = inds[k];
        for (size_t l = 0; l < ind.size(); l++)
        {
            std::string token = ind[l];
            if (index_map.count(token))
                continue;
            index_map[token] = cyclops_index_set[index++];
        }
    }

    std::vector<std::string> cyclops_inds;
    for (size_t k = 0L; k < inds.size(); k++)
    {
        std::stringstream ss;
        const std::vector<std::string> &ind = inds[k];
        for (size_t l = 0; l < ind.size(); l++)
        {
            ss << index_map[ind[l]];
        }
        cyclops_inds.push_back(ss.str());
    }
    return cyclops_inds;
}
}

int initialize(int argc, char **argv)
{
    MPI_Initialized(&details::initialized_mpi);

    if (!details::initialized_mpi)
    {
        int error = MPI_Init(&argc, &argv);
        if (error != MPI_SUCCESS)
        {
            throw std::runtime_error(
                "cyclops::initialize: Unable to initialize MPI.");
        }
    }

// I don't know how to initialize elemental with a different communicator.
#if defined(HAVE_ELEMENTAL)
    El::Initialize(argc, argv);
#endif

    globals::communicator = MPI_COMM_WORLD;

    MPI_Comm_rank(globals::communicator, &settings::rank);
    MPI_Comm_size(globals::communicator, &settings::nprocess);

    details::world = new CTF_World(globals::communicator, argc, argv);

    if (settings::debug)
    {
        print("Cyclops interface initialized. nprocess: %d\n",
              settings::nprocess);
    }

    return 0;
}

int initialize(MPI_Comm comm, int argc, char **argv)
{
    // Since we're provided a communicator ensure MPI is initialize
    MPI_Initialized(&details::initialized_mpi);

    if (!details::initialized_mpi)
    {
        throw std::runtime_error("cyclops::initialize: Communicator provided "
                                 "but MPI is not initialized.");
    }

    globals::communicator = comm;

#if defined(HAVE_ELEMENTAL)
    print("cycops::initialize: I don't know how to initialize Elemental with a "
          "different communicator.");
    El::Initialize(argc, argv);
#endif

    MPI_Comm_rank(globals::communicator, &settings::rank);
    MPI_Comm_size(globals::communicator, &settings::nprocess);

    details::world = new CTF_World(globals::communicator, argc, argv);

    if (settings::debug)
    {
        print("Cyclops interface initialized. nprocess: %d\n",
              settings::nprocess);
    }

    return 0;
}

void finalize()
{
    delete details::world;
    details::world = nullptr;

#if defined(HAVE_ELEMENTAL)
    El::Finalize();
#endif

    // if we initialized MPI then we finalize it.
    if (!details::initialized_mpi)
        MPI_Finalize();
}

CyclopsTensorImpl::CyclopsTensorImpl(const std::string &name,
                                     const Dimension &dims)
    : TensorImpl(kDistributed, name, dims)
{
    if (dims.size() == 0)
    {
        cyclops_ = new CTF_Scalar(0.0, *details::world);
        return;
    }

    int *local_sym = new int[dims.size()];
    std::fill(local_sym, local_sym + dims.size(), NS);
    int *local_dims = new int[dims.size()];
    std::copy(dims.begin(), dims.end(), local_dims);

    cyclops_ =
        new CTF_Tensor(dims.size(), local_dims, local_sym, *details::world);

    delete[] local_dims;
    delete[] local_sym;
}

CyclopsTensorImpl::~CyclopsTensorImpl() { delete cyclops_; }

double CyclopsTensorImpl::norm(int type) const
{
    switch (type)
    {
    case 0:
        return cyclops_->norm_infty();
    case 1:
        return cyclops_->norm1();
    case 2:
        return cyclops_->norm2();
    default:
        throw std::runtime_error("Unknown norm type passed to Cyclops tensor.");
    }
}

std::tuple<double, std::vector<size_t>> CyclopsTensorImpl::max() const
{
    double element_value;
    std::vector<size_t> element_indices;

    element_value = std::numeric_limits<double>::lowest();

    citerate([&](const std::vector<size_t> &indices, const double &value)
             {
                 if (value > element_value)
                 {
                     element_value = value;
                     element_indices = indices;
                 }
             });

    barrier();

    // Need to perform broadcast of smallest.
    size_t *myindices = new size_t[rank()];
    size_t *allindices = new size_t[rank() * ambit::settings::nprocess];
    double *allvalues = new double[ambit::settings::nprocess];

    // Broadcast each nodes smallest value to all nodes
    MPI_Allgather(&element_value, 1, MPI_DOUBLE, allvalues, 1, MPI_DOUBLE,
                  globals::communicator);

    for (int i = 0; i < rank(); i++)
        myindices[i] = element_indices[i];

    // Broadcast each nodes set of indices to all nodes
    MPI_Allgather(myindices, rank() * sizeof(size_t), MPI_CHAR, allindices,
                  rank() * sizeof(size_t), MPI_CHAR, globals::communicator);

    // Now that we have each nodes' minimum value check each for the global
    // minimum
    for (size_t i = 0; i < ambit::settings::nprocess; i++)
    {
        if (allvalues[i] > element_value)
        {
            element_value = allvalues[i];
            for (size_t j = 0; j < rank(); j++)
                element_indices[j] = allindices[rank() * i + j];
        }
    }

    delete[] allvalues;
    delete[] allindices;
    delete[] myindices;

    barrier();

    std::tuple<double, std::vector<size_t>> result;
    std::get<0>(result) = element_value;
    std::get<1>(result) = element_indices;

    return result;
}

std::tuple<double, std::vector<size_t>> CyclopsTensorImpl::min() const
{
    double element_value;
    std::vector<size_t> element_indices;

    element_value = std::numeric_limits<double>::max();

    citerate([&](const std::vector<size_t> &indices, const double &value)
             {
                 if (value < element_value)
                 {
                     element_value = value;
                     element_indices = indices;
                 }
             });

    barrier();

    // Need to perform broadcast of smallest.
    size_t *myindices = new size_t[rank()];
    size_t *allindices = new size_t[rank() * ambit::settings::nprocess];
    double *allvalues = new double[ambit::settings::nprocess];

    // Broadcast each nodes smallest value to all nodes
    MPI_Allgather(&element_value, 1, MPI_DOUBLE, allvalues, 1, MPI_DOUBLE,
                  globals::communicator);

    for (int i = 0; i < rank(); i++)
        myindices[i] = element_indices[i];

    // Broadcast each nodes set of indices to all nodes
    MPI_Allgather(myindices, rank() * sizeof(size_t), MPI_CHAR, allindices,
                  rank() * sizeof(size_t), MPI_CHAR, globals::communicator);

    // Now that we have each nodes' minimum value check each for the global
    // minimum
    for (size_t i = 0; i < ambit::settings::nprocess; i++)
    {
        if (allvalues[i] < element_value)
        {
            element_value = allvalues[i];
            for (size_t j = 0; j < rank(); j++)
                element_indices[j] = allindices[rank() * i + j];
        }
    }

    delete[] allvalues;
    delete[] allindices;
    delete[] myindices;

    barrier();

    std::tuple<double, std::vector<size_t>> result;
    std::get<0>(result) = element_value;
    std::get<1>(result) = element_indices;

    return result;
}

void CyclopsTensorImpl::scale(double beta)
{
    if (beta == 0.0)
    {
        *cyclops_ = 0.0;
    }
    else
    {
        long_int local_size;
        double *local_data;

        local_data = cyclops_->get_raw_data(&local_size);

        VECTORIZED_LOOP
        for (long_int i = 0; i < local_size; ++i)
        {
            local_data[i] *= beta;
        }
    }
}

void CyclopsTensorImpl::set(double alpha)
{
    long_int local_size;
    double *local_data;

    local_data = cyclops_->get_raw_data(&local_size);

    VECTORIZED_LOOP
    for (long_int i = 0; i < local_size; ++i)
    {
        local_data[i] = alpha;
    }
}

void CyclopsTensorImpl::permute(ConstTensorImplPtr A,
                                const std::vector<std::string> &Cinds,
                                const std::vector<std::string> &Ainds,
                                double alpha, double beta)
{
    GET_CTF_TENSOR(A);

    std::vector<std::string> cyclops_inds =
        generateCyclopsLabels({Cinds, Ainds});
    std::string Ccyclops = cyclops_inds[0];
    std::string Acyclops = cyclops_inds[1];

    scale(beta);
    (*cyclops_)[Ccyclops.c_str()] += alpha * (*tA)[Acyclops.c_str()];
}

void CyclopsTensorImpl::contract(ConstTensorImplPtr A, ConstTensorImplPtr B,
                                 const std::vector<std::string> &Cinds,
                                 const std::vector<std::string> &Ainds,
                                 const std::vector<std::string> &Binds,
                                 double alpha, double beta)
{
    GET_CTF_TENSOR(A);
    GET_CTF_TENSOR(B);

    std::vector<std::string> cyclops_inds =
        generateCyclopsLabels({Cinds, Ainds, Binds});
    std::string Ccyclops = cyclops_inds[0];
    std::string Acyclops = cyclops_inds[1];
    std::string Bcyclops = cyclops_inds[2];

    scale(beta);
    (*cyclops_)[Ccyclops.c_str()] +=
        alpha * (*tA)[Acyclops.c_str()] * (*tB)[Bcyclops.c_str()];
}

std::map<std::string, TensorImplPtr>
CyclopsTensorImpl::syev(EigenvalueOrder order) const
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
    El::DistMatrix<double, El::VR, El::STAR> w;
    El::DistMatrix<double> X;
    El::SortType sort = order == kAscending ? El::ASCENDING : El::DESCENDING;
    El::HermitianEig(El::LOWER, H, w, X, sort);

    // construct cyclops tensors to hold eigen vectors and values.
    Dimension value_dims(1);
    value_dims[0] = length;
    CyclopsTensorImpl *vectors = new CyclopsTensorImpl("Eigenvectors", dims());
    CyclopsTensorImpl *values =
        new CyclopsTensorImpl("Eigenvalues", value_dims);

    // populate cyclops with elemental data
    vectors->copyFromElemental2(X);
    values->copyFromElemental1(w);

    std::map<std::string, TensorImplPtr> results;
    results["eigenvalues"] = values;
    results["eigenvectors"] = vectors;

    return results;
#else

    ThrowNotImplementedException;

#endif
}

TensorImplPtr CyclopsTensorImpl::power(double alpha, double condition) const
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
    El::DistMatrix<double, El::VR, El::STAR> w;
    El::DistMatrix<double> X;
    El::SortType sort = El::ASCENDING;
    El::HermitianEig(El::LOWER, H, w, X, sort);

    const int numLocalEigs = w.LocalHeight();
    double maxLocalEig = 0.0;
    for (int iLoc = 0; iLoc < numLocalEigs; ++iLoc)
    {
        const double omega = w.GetLocal(iLoc, 0);
        maxLocalEig = std::max(maxLocalEig, omega);
    }
    const double maxEig =
        El::mpi::AllReduce(maxLocalEig, El::mpi::MAX, El::mpi::COMM_WORLD);

    // Overwrite the eigenvalues with f(w)
    for (int iLoc = 0; iLoc < numLocalEigs; ++iLoc)
    {
        const double omega = w.GetLocal(iLoc, 0);
        double new_omega = 0.0;

        if (alpha < 0.0 && fabs(omega) < condition * maxEig)
            new_omega = 0.0;
        else
        {
            new_omega = pow(omega, alpha);
            if (!std::isfinite(new_omega))
                new_omega = 0.0;
        }

        w.SetLocal(iLoc, 0, new_omega);
    }

    El::HermitianFromEVD(El::LOWER, H, w, X);

    CyclopsTensorImpl *result =
        new CyclopsTensorImpl(name() + "^" + std::to_string(alpha), dims());
    result->copyFromLowerElementalToFull2(H);
    return result;
#else
    ThrowNotImplementedException;
#endif
}

void CyclopsTensorImpl::iterate(
    const std::function<void(const std::vector<size_t> &, double &)> &func)
{
    std::vector<size_t> indices(rank(), 0);
    std::vector<size_t> addressing(rank(), 1);

    // form addressing array
    for (size_t n = 1; n < rank(); ++n)
    {
        addressing[n] = addressing[n - 1] * dim(n - 1);
    }

    long_int nelem;
    size_t nrank = rank();
    kv_pair *pairs;

    cyclops_->read_local(&nelem, &pairs);
    for (size_t n = 0; n < nelem; ++n)
    {
        size_t d = pairs[n].k;
        for (int k = nrank - 1; k >= 0; --k)
        {
            indices[k] = d / addressing[k];
            d = d % addressing[k];
        }

        func(indices, pairs[n].d);
    }

    // the user may have modified the data, must write to the tensor
    cyclops_->write(nelem, pairs);

    free(pairs);
}

void CyclopsTensorImpl::citerate(
    const std::function<void(const std::vector<size_t> &, const double &)>
        &func) const
{
    std::vector<size_t> indices(rank(), 0);
    std::vector<size_t> addressing(rank(), 1);

    // form addressing array
    for (size_t n = 1; n < rank(); ++n)
    {
        addressing[n] = addressing[n - 1] * dim(n - 1);
    }

    long_int nelem;
    size_t nrank = rank();
    kv_pair *pairs;

    cyclops_->read_local(&nelem, &pairs);
    for (size_t n = 0; n < nelem; ++n)
    {
        size_t d = pairs[n].k;
        for (int k = nrank - 1; k >= 0; --k)
        {
            indices[k] = d / addressing[k];
            d = d % addressing[k];
        }

        func(indices, pairs[n].d);
    }

    free(pairs);
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
    for (int i = 0; i < U.LocalHeight(); i++)
    {
        for (int j = 0; j < U.LocalWidth(); j++)
        {
            const int c = cshift + i * cstride;
            const int r = rshift + j * rstride;

            pairs.push_back(kv_pair(r * cols + c, 0));
        }
    }

    // gather data from cyclops
    cyclops_->read(pairs.size(), pairs.data());

    // populate elemental
    for (size_t p = 0; p < pairs.size(); p++)
    {
        const int r = pairs[p].k / cols;
        const int c = pairs[p].k - cols * r;
        const int i = (c - cshift) / cstride;
        const int j = (r - rshift) / rstride;

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

    for (int i = 0; i < X.LocalHeight(); i++)
    {
        for (int j = 0; j < X.LocalWidth(); j++)
        {
            const int c = cshift + i * cstride;
            const int r = rshift + j * rstride;

            pairs.push_back(kv_pair(c * length + r, X.GetLocal(i, j)));
        }
    }
    cyclops_->write(pairs.size(), pairs.data());
}

void CyclopsTensorImpl::copyFromLowerElementalToFull2(
    const El::DistMatrix<double> &X)
{
    size_t length = dims()[1];
    std::vector<kv_pair> pairs;
    const int cshift = X.ColShift();
    const int rshift = X.RowShift();
    const int cstride = X.ColStride();
    const int rstride = X.RowStride();

    for (int i = 0; i < X.LocalHeight(); i++)
    {
        for (int j = 0; j < X.LocalWidth(); j++)
        {
            const int c = cshift + i * cstride;
            const int r = rshift + j * rstride;

            // lower triangle
            if (r < c)
            {
                pairs.push_back(kv_pair(r * length + c, X.GetLocal(i, j)));
                pairs.push_back(kv_pair(c * length + r, X.GetLocal(i, j)));
            }
            // diagonal
            else if (c == r)
            {
                pairs.push_back(kv_pair(r * length + c, X.GetLocal(i, j)));
            }
        }
    }
    cyclops_->write(pairs.size(), pairs.data());
}

void CyclopsTensorImpl::copyFromElemental1(
    const El::DistMatrix<double, El::VR, El::STAR> &w)
{
    std::vector<kv_pair> pairs;
    const int cshift = w.ColShift();
    const int cstride = w.ColStride();
    for (int i = 0; i < w.LocalHeight(); i++)
    {
        pairs.push_back(kv_pair(cshift + i * cstride, w.GetLocal(i, 0)));
    }
    cyclops_->write(pairs.size(), pairs.data());
}
#endif
}
}
