#include "tensorimpl.h"
#include "core.h"
#include "memory.h"
#include "math/math.h"
#include "indices.h"
#include <string.h>
#include <cmath>

#include <boost/timer/timer.hpp>

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
    C_DSCAL(numel(), a, data_, 1);
}

double CoreTensorImpl::norm(double /*power*/) const
{
    ThrowNotImplementedException;
}
double CoreTensorImpl::rms(double /*power*/) const
{
    ThrowNotImplementedException;
}

void CoreTensorImpl::scale_and_add(const double& a, ConstTensorImplPtr x)
{
    if (numel() != x->numel()) {
        throw std::runtime_error("Tensors must have the same number of elements.");
    }

    C_DAXPY(numel(),
            a,
            ((ConstCoreTensorImplPtr)x)->data(),
            1,
            data_,
            1);
}

void CoreTensorImpl::pointwise_multiplication(ConstTensorImplPtr x)
{
    if (numel() != x->numel()) {
        throw std::runtime_error("Tensors must have the same number of elements.");
    }

    const double* rhs = ((ConstCoreTensorImplPtr)x)->data();
    OMP_VECTORIZED_STATIC_LOOP
    for (size_t ind=0, end=numel(); ind < end; ++ind) {
        data_[ind] *= rhs[ind];
    }
}

void CoreTensorImpl::pointwise_division(ConstTensorImplPtr x)
{
    if (numel() != x->numel()) {
        throw std::runtime_error("Tensors must have the same number of elements.");
    }

    const double* rhs = ((ConstCoreTensorImplPtr)x)->data();
    OMP_VECTORIZED_STATIC_LOOP
    for (size_t ind=0, end=numel(); ind < end; ++ind) {
        data_[ind] /= rhs[ind];
    }
}

double CoreTensorImpl::dot(ConstTensorImplPtr x) const
{
    if (numel() != x->numel()) {
        throw std::runtime_error("Tensors must have the same number of elements.");
    }
    dimensionCheck(this, x, true);

    return C_DDOT(numel(), data_, 1, ((ConstCoreTensorImplPtr)x)->data(), 1);
}

void CoreTensorImpl::contract(
    ConstTensorImplPtr A,
    ConstTensorImplPtr B,
    const Indices& Cinds,
    const Indices& Ainds,
    const Indices& Binds,
    double alpha,
    double beta)
{
    TensorImplPtr C = this;

    // => Permutation Logic <= //

    // Determine unique indices
    std::vector<std::string> inds;
    inds.insert(inds.end(),Cinds.begin(),Cinds.end());
    inds.insert(inds.end(),Ainds.begin(),Ainds.end());
    inds.insert(inds.end(),Binds.begin(),Binds.end());
    std::sort(inds.begin(), inds.end());
    std::vector<std::string>::iterator it = std::unique(inds.begin(), inds.end());
    inds.resize(std::distance(inds.begin(),it));

    // Determine index types and positions (and GEMM sizes while we are there)
    size_t ABC_size = 1L;
    size_t BC_size = 1L;
    size_t AC_size = 1L;
    size_t AB_size = 1L;
    std::vector<std::string> compound_names = {"PC", "PA", "PB", "iC", "iA", "jC", "jB", "kA", "kB"};
    std::map<std::string, std::vector<std::pair<int, std::string>>> compound_inds;
    for (size_t ind = 0L; ind < compound_names.size(); ind++) {
        compound_inds[compound_names[ind]] = std::vector<std::pair<int, std::string>>();
    }
    for (size_t ind = 0L; ind < inds.size(); ind++) {
        std::string index = inds[ind];
        int Cpos = indices::find_index_in_vector(Cinds,index);
        int Apos = indices::find_index_in_vector(Ainds,index);
        int Bpos = indices::find_index_in_vector(Binds,index);
        if (Cpos != -1 && Apos != -1 && Bpos != -1) {
            if (C->dims()[Cpos] != A->dims()[Apos] || C->dims()[Cpos] != B->dims()[Bpos])
                throw std::runtime_error("Invalid ABC (Hadamard) index size");
            compound_inds["PC"].push_back(std::make_pair(Cpos,index));
            compound_inds["PA"].push_back(std::make_pair(Apos,index));
            compound_inds["PB"].push_back(std::make_pair(Bpos,index));
            ABC_size *= C->dims()[Cpos];
        } else if (Cpos != -1 && Apos != -1 && Bpos == -1) {
            if (C->dims()[Cpos] != A->dims()[Apos])
                throw std::runtime_error("Invalid AC (Left) index size");
            compound_inds["iC"].push_back(std::make_pair(Cpos,index));
            compound_inds["iA"].push_back(std::make_pair(Apos,index));
            AC_size *= C->dims()[Cpos];
        } else if (Cpos != -1 && Apos == -1 && Bpos != -1) {
            if (C->dims()[Cpos] != B->dims()[Bpos])
                throw std::runtime_error("Invalid BC (Right) index size");
            compound_inds["jC"].push_back(std::make_pair(Cpos,index));
            compound_inds["jB"].push_back(std::make_pair(Bpos,index));
            BC_size *= C->dims()[Cpos];
        } else if (Cpos == -1 && Apos != -1 && Bpos != -1) {
            if (A->dims()[Apos] != B->dims()[Bpos])
                throw std::runtime_error("Invalid AB (Contraction) index size");
            compound_inds["kA"].push_back(std::make_pair(Apos,index));
            compound_inds["kB"].push_back(std::make_pair(Bpos,index));
            AB_size *= B->dims()[Bpos];
        } else {
            throw std::runtime_error("Invalid contraction topology - index only occurs once.");
        }
    }

    // Sort compound indices by primitive indices to determine continuity (will be sequential if continuous)
    for (size_t ind = 0L; ind < compound_names.size(); ind++) {
        std::sort(compound_inds[compound_names[ind]].begin(), compound_inds[compound_names[ind]].end());
    }

    // Flags to mark if tensors must be permuted
    bool permC = false;
    bool permA = false;
    bool permB = false;

    // Contiguous Index Test (always requires permutation)
    permC = permC || !indices::contiguous(compound_inds["PC"]);
    permC = permC || !indices::contiguous(compound_inds["iC"]);
    permC = permC || !indices::contiguous(compound_inds["jC"]);
    permA = permA || !indices::contiguous(compound_inds["PA"]);
    permA = permA || !indices::contiguous(compound_inds["iA"]);
    permA = permA || !indices::contiguous(compound_inds["kA"]);
    permB = permB || !indices::contiguous(compound_inds["PB"]);
    permB = permB || !indices::contiguous(compound_inds["jB"]);
    permB = permB || !indices::contiguous(compound_inds["kB"]);

    // Hadamard Test (always requires permutation)
    int Psize = compound_inds["PC"].size();
    if (Psize) {
        permC = permC || (compound_inds["PC"][0].first != 0);
        permA = permA || (compound_inds["PA"][0].first != 0);
        permB = permB || (compound_inds["PB"][0].first != 0);
    }

    /// Figure out the initial transposes (will be fixed if perm is set)
    bool A_transpose = false;
    bool B_transpose = false;
    bool C_transpose = false;
    if (compound_inds["iC"].size() && compound_inds["iC"][0].first != Psize) C_transpose = true;
    if (compound_inds["iA"].size() && compound_inds["iA"][0].first != Psize) A_transpose = true;
    if (compound_inds["jB"].size() && compound_inds["jB"][0].first == Psize) B_transpose = true;

    // Fix contiguous considerations (already in correct order for contiguous cases)
    std::map<std::string, std::vector<std::string> > compound_inds2;
    for (size_t ind = 0L; ind < compound_names.size(); ind++) {
        std::string key = compound_names[ind];
        std::vector<std::string> vals;
        for (size_t ind2 = 0L; ind2 < compound_inds[key].size(); ind2++) {
            vals.push_back(compound_inds[key][ind2].second);
        }
        compound_inds2[key] = vals;
    }

    /**
    * Fix permutation order considerations
    *
    * Rules if a permutation mismatch is detected:
    * -If both tensors are already on the permute list, it doesn't matter which is fixed
    * -Else if one tensor is already on the permute list but not the other, fix the one that is already on the permute list
    * -Else fix the smaller tensor
    *
    * Note: this scheme is not optimal is permutation mismatches exist in P - for reasons of simplicity, A and B are
    * permuted to C's P order, with no present considerations of better pathways
    **/
    if (!indices::equivalent(compound_inds2["iC"],compound_inds2["iA"])) {
        if (permC) {
            compound_inds2["iC"] = compound_inds2["iA"];
        } else if (permA) {
            compound_inds2["iA"] = compound_inds2["iC"];
        } else if (C->numel() <= A->numel()) {
            compound_inds2["iC"] = compound_inds2["iA"];
            permC = true;
        } else {
            compound_inds2["iA"] = compound_inds2["iC"];
            permA = true;
        }
    }
    if (!indices::equivalent(compound_inds2["jC"],compound_inds2["jB"])) {
        if (permC) {
            compound_inds2["jC"] = compound_inds2["jB"];
        } else if (permB) {
            compound_inds2["jB"] = compound_inds2["jC"];
        } else if (C->numel() <= B->numel()) {
            compound_inds2["jC"] = compound_inds2["jB"];
            permC = true;
        } else {
            compound_inds2["jB"] = compound_inds2["jC"];
            permB = true;
        }
    }
    if (!indices::equivalent(compound_inds2["kA"],compound_inds2["kB"])) {
        if (permA) {
            compound_inds2["kA"] = compound_inds2["kB"];
        } else if (permB) {
            compound_inds2["kB"] = compound_inds2["kA"];
        } else if (A->numel() <= B->numel()) {
            compound_inds2["kA"] = compound_inds2["kB"];
            permA = true;
        } else {
            compound_inds2["kB"] = compound_inds2["kA"];
            permB = true;
        }
    }
    if (!indices::equivalent(compound_inds2["PC"],compound_inds2["PA"])) {
        compound_inds2["PA"] = compound_inds2["PC"];
        permA = true;
    }
    if (!indices::equivalent(compound_inds2["PC"],compound_inds2["PB"])) {
        compound_inds2["PB"] = compound_inds2["PC"];
        permB = true;
    }

    /// Assign the permuted indices (if flagged for permute) or the original indices
    std::vector<std::string> Cinds2;
    std::vector<std::string> Ainds2;
    std::vector<std::string> Binds2;
    if (permC) {
        Cinds2.insert(Cinds2.end(),compound_inds2["PC"].begin(),compound_inds2["PC"].end());
        Cinds2.insert(Cinds2.end(),compound_inds2["iC"].begin(),compound_inds2["iC"].end());
        Cinds2.insert(Cinds2.end(),compound_inds2["jC"].begin(),compound_inds2["jC"].end());
        C_transpose = false;
    } else {
        Cinds2 = Cinds;
    }
    if (permA) {
        Ainds2.insert(Ainds2.end(),compound_inds2["PA"].begin(),compound_inds2["PA"].end());
        Ainds2.insert(Ainds2.end(),compound_inds2["iA"].begin(),compound_inds2["iA"].end());
        Ainds2.insert(Ainds2.end(),compound_inds2["kA"].begin(),compound_inds2["kA"].end());
        A_transpose = false;
    } else {
        Ainds2 = Ainds;
    }
    if (permB) {
        Binds2.insert(Binds2.end(),compound_inds2["PB"].begin(),compound_inds2["PB"].end());
        Binds2.insert(Binds2.end(),compound_inds2["jB"].begin(),compound_inds2["jB"].end());
        Binds2.insert(Binds2.end(),compound_inds2["kB"].begin(),compound_inds2["kB"].end());
        B_transpose = true;
    } else {
        Binds2 = Binds;
    }

    // So what exactly happened?
    /**
    printf("==> Core Contraction <==\n\n");
    printf("Original: C[");
    for (size_t ind = 0l; ind < Cinds.size(); ind++) {
        printf("%s", Cinds[ind].c_str());
    }
    printf("] = A[");
    for (size_t ind = 0l; ind < Ainds.size(); ind++) {
        printf("%s", Ainds[ind].c_str());
    }
    printf("] * B[");
    for (size_t ind = 0l; ind < Binds.size(); ind++) {
        printf("%s", Binds[ind].c_str());
    }
    printf("]\n");
    printf("New:      C[");
    for (size_t ind = 0l; ind < Cinds2.size(); ind++) {
        printf("%s", Cinds2[ind].c_str());
    }
    printf("] = A[");
    for (size_t ind = 0l; ind < Ainds2.size(); ind++) {
        printf("%s", Ainds2[ind].c_str());
    }
    printf("] * B[");
    for (size_t ind = 0l; ind < Binds2.size(); ind++) {
        printf("%s", Binds2[ind].c_str());
    }
    printf("]\n");
    printf("C Permuted: %s\n", permC ? "Yes" : "No");
    printf("A Permuted: %s\n", permA ? "Yes" : "No");
    printf("B Permuted: %s\n", permB ? "Yes" : "No");
    printf("\n");
    **/

    // => Alias or Allocate A, B, C <= //

    Dimension Cdims2 = indices::permuted_dimension(C->dims(), Cinds2, Cinds);
    Dimension Adims2 = indices::permuted_dimension(A->dims(), Ainds2, Ainds);
    Dimension Bdims2 = indices::permuted_dimension(B->dims(), Binds2, Binds);

    double* Cp = ((CoreTensorImplPtr)C)->data();
    double* Ap = ((CoreTensorImplPtr)A)->data();
    double* Bp = ((CoreTensorImplPtr)B)->data();
    double* C2p = Cp;
    double* A2p = Ap;
    double* B2p = Bp;

    /// TODO: This is ugly. Overall, where do we use shared pointers, references, const references, or object copy?
    shared_ptr<CoreTensorImpl> C2;
    shared_ptr<CoreTensorImpl> B2;
    shared_ptr<CoreTensorImpl> A2;
    if (permC) {
        C2 = shared_ptr<CoreTensorImpl>(new CoreTensorImpl("C2", Cdims2));
        C2p = C2->data();
    }
    if (permA) {
        A2 = shared_ptr<CoreTensorImpl>(new CoreTensorImpl("A2", Adims2));
        A2p = A2->data();
    }
    if (permB) {
        B2 = shared_ptr<CoreTensorImpl>(new CoreTensorImpl("B2", Bdims2));
        B2p = B2->data();
    }

    // => Permute A, B, and C if Necessary <= //

    if (permC) C2->permute(C,Cinds2,Cinds);
    if (permA) A2->permute(A,Ainds2,Ainds);
    if (permB) B2->permute(B,Binds2,Binds);

    // => GEMM Indexing <= //

    // => GEMM <= //

    for (size_t P = 0L; P < ABC_size; P++) {

        char transL;
        char transR;
        size_t nrow;
        size_t ncol;
        double* Lp;
        double* Rp;
        size_t ldaL;
        size_t ldaR;

        if (C_transpose) {
            Lp = B2p;
            Rp = A2p;
            nrow = BC_size;
            ncol = AC_size;
            transL = (B_transpose ? 'N' : 'T');
            transR = (A_transpose ? 'N' : 'T');
            ldaL = (B_transpose ? AB_size : BC_size);
            ldaR = (A_transpose ? AC_size : AB_size);
        } else {
            Lp = A2p;
            Rp = B2p;
            nrow = AC_size;
            ncol = BC_size;
            transL = (A_transpose ? 'T' : 'N');
            transR = (B_transpose ? 'T' : 'N');
            ldaL = (A_transpose ? AC_size : AB_size);
            ldaR = (B_transpose ? AB_size : BC_size);
        }

        size_t nzip = AB_size;
        size_t ldaC = (C_transpose ? AC_size : BC_size);

        C_DGEMM(transL,transR,nrow,ncol,nzip,alpha,Lp,ldaL,Rp,ldaR,beta,C2p,ldaC);

        C2p += AC_size * BC_size;
        A2p += AB_size * AC_size;
        B2p += AB_size * BC_size;
    }

    // => Permute C if Necessary <= //

    if (permC) C->permute(C2.get(),Cinds,Cinds2);
}

void CoreTensorImpl::permute(
    ConstTensorImplPtr A,
    const Indices& CindsS,
    const Indices& AindsS,
    double alpha,
    double beta)
{
    // => Convert to indices of A <= //

    std::vector<size_t> Ainds = indices::permutation_order(CindsS, AindsS);
    for (size_t dim = 0; dim < rank(); dim++) {
        if (dims()[dim] != A->dims()[Ainds[dim]])
            throw std::runtime_error("Permuted tensors do not have same dimensions");
    }

    /// Data pointers
    double* Cp = data();
    double* Ap = ((const CoreTensorImplPtr)A)->data();

    /// Beta scale 
    C_DSCAL(numel(),beta,Cp,1);

    // => Index Logic <= //

    /// Determine the number of united fast indices and memcpy size
    /// C_ij = A_ji would have no fast dimensions and a fast size of 1
    /// C_ijk = A_jik would have k as a fast dimension, and a fast size of dim(k)
    /// C_ijkl = A_jikl would have k and l as fast dimensions, and a fast size of dim(k) * dim(l)
    int fast_dims = 0;
    size_t fast_size = 1L;
    for (int dim = ((int)rank()) - 1; dim >= 0; dim--) {
        if (dim == ((int)Ainds[dim])) {
            fast_dims++;
            fast_size *= dims()[dim];
        } else {
            break;
        }
    }

    /// Determine the total number of memcpy operations
    int slow_dims = rank() - fast_dims;

    /// Fully sorted case or (equivalently) rank-0 or rank-1 tensors
    if (slow_dims == 0) {
        //::memcpy(Cp,Ap,sizeof(double)*fast_size);
        C_DAXPY(fast_size,alpha,Ap,1,Cp,1);
        return;
    }

    assert(slow_dims > 1); // slow_dims != 1

    /// Number of collapsed indices in permutation traverse
    size_t slow_size = 1L;
    for (int dim = 0; dim < slow_dims; dim++) {
        slow_size *= dims()[dim];
    }

    /// Strides of slow indices of tensor A in its own ordering
    std::vector<size_t> Astrides(slow_dims,0L);
    if (slow_dims != 0) Astrides[slow_dims-1] = fast_size;
    for (int dim = slow_dims-2; dim >= 0; dim--) {
        Astrides[dim] = Astrides[dim+1] * A->dims()[dim+1];
    }

    /// Strides of slow indices of tensor A in the ordering of tensor C
    std::vector<size_t> AstridesC(slow_dims,0L);
    for (int dim = 0; dim < slow_dims; dim++) {
        AstridesC[dim] = Astrides[Ainds[dim]];
    }

    /// Strides of slow indices of tensor C in its own ordering
    std::vector<size_t> Cstrides(slow_dims,0L);
    if (slow_dims != 0) Cstrides[slow_dims-1] = fast_size;
    for (int dim = slow_dims-2; dim >= 0; dim--) {
        Cstrides[dim] = Cstrides[dim+1] * dims()[dim+1];
    }

    /// Handle to dimensions of C
    const std::vector<size_t>& Csizes = dims();

    // => Permute Operation <= //

    if (slow_dims == 2) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
            double* Atp = Ap +
                Cind0 * AstridesC[0] +
                Cind1 * AstridesC[1];
            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
            Ctp += fast_size;
        }}
    } else if (slow_dims == 3) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
            double* Atp = Ap +
                Cind0 * AstridesC[0] +
                Cind1 * AstridesC[1] +
                Cind2 * AstridesC[2];
            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
            Ctp += fast_size;
        }}}
    } else if (slow_dims == 4) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
            double* Atp = Ap +
                Cind0 * AstridesC[0] +
                Cind1 * AstridesC[1] +
                Cind2 * AstridesC[2] +
                Cind3 * AstridesC[3];
            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
            Ctp += fast_size;
        }}}}
    } else if (slow_dims == 5) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
        for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++) {
            double* Atp = Ap +
                Cind0 * AstridesC[0] +
                Cind1 * AstridesC[1] +
                Cind2 * AstridesC[2] +
                Cind3 * AstridesC[3] +
                Cind4 * AstridesC[4];
            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
            Ctp += fast_size;
        }}}}}
    } else if (slow_dims == 6) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
        for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++) {
        for (size_t Cind5 = 0L; Cind5 < Csizes[5]; Cind5++) {
            double* Atp = Ap +
                Cind0 * AstridesC[0] +
                Cind1 * AstridesC[1] +
                Cind2 * AstridesC[2] +
                Cind3 * AstridesC[3] +
                Cind4 * AstridesC[4] +
                Cind5 * AstridesC[5];
            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
            Ctp += fast_size;
        }}}}}}
    } else if (slow_dims == 7) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
        for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++) {
        for (size_t Cind5 = 0L; Cind5 < Csizes[5]; Cind5++) {
        for (size_t Cind6 = 0L; Cind6 < Csizes[6]; Cind6++) {
            double* Atp = Ap +
                Cind0 * AstridesC[0] +
                Cind1 * AstridesC[1] +
                Cind2 * AstridesC[2] +
                Cind3 * AstridesC[3] +
                Cind4 * AstridesC[4] +
                Cind5 * AstridesC[5] +
                Cind6 * AstridesC[6];
            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
            Ctp += fast_size;
        }}}}}}}
    } else if (slow_dims == 8) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
        for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++) {
        for (size_t Cind5 = 0L; Cind5 < Csizes[5]; Cind5++) {
        for (size_t Cind6 = 0L; Cind6 < Csizes[6]; Cind6++) {
        for (size_t Cind7 = 0L; Cind7 < Csizes[7]; Cind7++) {
            double* Atp = Ap +
                Cind0 * AstridesC[0] +
                Cind1 * AstridesC[1] +
                Cind2 * AstridesC[2] +
                Cind3 * AstridesC[3] +
                Cind4 * AstridesC[4] +
                Cind5 * AstridesC[5] +
                Cind6 * AstridesC[6] +
                Cind7 * AstridesC[7];
            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
            Ctp += fast_size;
        }}}}}}}}
    } else {
        #pragma omp parallel for
        for (size_t ind = 0L; ind < slow_size; ind++) {
            double* Ctp = Cp + ind * fast_size;
            double* Atp = Ap;
            size_t num = ind;
            for (int dim = slow_dims - 1; dim >= 0; dim--) {
                size_t val = num % Csizes[dim]; // value of the dim-th index
                num /= Csizes[dim];
                Atp += val * AstridesC[dim];
            }
            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }
    }
}
void CoreTensorImpl::slice(
    ConstTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha,
    double beta)
{
    TensorImplPtr C = this;

    /// Data pointers
    double* Cp = data();
    double* Ap = ((const CoreTensorImplPtr)A)->data();

    // => Index Logic <= //

    // TODO Validity checks
    // TODO This is not valid for rank() == 0
    
    /// Sizes of stripes
    std::vector<size_t> sizes(rank(),0L);
    for (size_t ind = 0L; ind < rank(); ind++) {
        sizes[ind] = Cinds[ind].second - Cinds[ind].first;
    }

    /// Size of contiguous DAXPY call
    int fast_dims = (rank() == 0 ? 0 : 1);
    size_t fast_size = (rank() == 0 ? 1L : sizes[rank() - 1]);
    for (int ind = ((int) rank()) - 2; ind >= 0; ind--) {
        if (sizes[ind+1] == A->dims()[ind+1] && sizes[ind+1] == C->dims()[ind+1]) {
            fast_dims++;
            fast_size *= sizes[ind];
        }
    }

    int slow_dims = rank() - fast_dims;
    size_t slow_size = 1L;
    for (int dim = 0; dim < slow_dims; dim++) {
        slow_size *= sizes[dim];
    }

    std::vector<size_t> Astrides(rank());
    Astrides[rank() - 1] = 1L;
    for (int ind = ((int)rank() - 2); ind >= 0; ind--) {
        Astrides[ind] = Astrides[ind+1] * A->dims()[ind+1];
    }

    std::vector<size_t> Cstrides(rank());
    Cstrides[rank() - 1] = 1L;
    for (int ind = ((int)rank() - 2); ind >= 0; ind--) {
        Cstrides[ind] = Cstrides[ind+1] * C->dims()[ind+1];
    }

    // => Slice Operation <= //

    if (slow_dims == 0) {
        double* Atp = Ap 
            + Ainds[slow_dims].first * Astrides[slow_dims]; 
        double* Ctp = Cp 
            + Cinds[slow_dims].first * Cstrides[slow_dims]; 
        C_DSCAL(fast_size,beta,Ctp,1);
        C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
    } else if (slow_dims == 1) {
        #pragma omp parallel for
        for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++) {
            double* Atp = Ap 
                + (Ainds[0].first + ind0) * Astrides[0]
                + Ainds[slow_dims].first * Astrides[slow_dims]; 
            double* Ctp = Cp 
                + (Cinds[0].first + ind0) * Cstrides[0]
                + Cinds[slow_dims].first * Cstrides[slow_dims]; 
            C_DSCAL(fast_size,beta,Ctp,1);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }
    } else if (slow_dims == 2) {
        #pragma omp parallel for
        for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++) {
        for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++) {
            double* Atp = Ap 
                + (Ainds[0].first + ind0) * Astrides[0]
                + (Ainds[1].first + ind1) * Astrides[1]
                + Ainds[slow_dims].first * Astrides[slow_dims]; 
            double* Ctp = Cp 
                + (Cinds[0].first + ind0) * Cstrides[0]
                + (Cinds[1].first + ind1) * Cstrides[1]
                + Cinds[slow_dims].first * Cstrides[slow_dims]; 
            C_DSCAL(fast_size,beta,Ctp,1);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }}
    } else if (slow_dims == 3) {
        #pragma omp parallel for
        for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++) {
        for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++) {
        for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++) {
            double* Atp = Ap 
                + (Ainds[0].first + ind0) * Astrides[0]
                + (Ainds[1].first + ind1) * Astrides[1]
                + (Ainds[2].first + ind2) * Astrides[2]
                + Ainds[slow_dims].first * Astrides[slow_dims]; 
            double* Ctp = Cp 
                + (Cinds[0].first + ind0) * Cstrides[0]
                + (Cinds[1].first + ind1) * Cstrides[1]
                + (Cinds[2].first + ind2) * Cstrides[2]
                + Cinds[slow_dims].first * Cstrides[slow_dims]; 
            C_DSCAL(fast_size,beta,Ctp,1);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }}}
    } else if (slow_dims == 4) {
        #pragma omp parallel for
        for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++) {
        for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++) {
        for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++) {
        for (size_t ind3 = 0L; ind3 < sizes[3]; ind3++) {
            double* Atp = Ap 
                + (Ainds[0].first + ind0) * Astrides[0]
                + (Ainds[1].first + ind1) * Astrides[1]
                + (Ainds[2].first + ind2) * Astrides[2]
                + (Ainds[3].first + ind3) * Astrides[3]
                + Ainds[slow_dims].first * Astrides[slow_dims]; 
            double* Ctp = Cp 
                + (Cinds[0].first + ind0) * Cstrides[0]
                + (Cinds[1].first + ind1) * Cstrides[1]
                + (Cinds[2].first + ind2) * Cstrides[2]
                + (Cinds[3].first + ind3) * Cstrides[3]
                + Cinds[slow_dims].first * Cstrides[slow_dims]; 
            C_DSCAL(fast_size,beta,Ctp,1);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }}}}
    } else if (slow_dims == 5) {
        #pragma omp parallel for
        for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++) {
        for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++) {
        for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++) {
        for (size_t ind3 = 0L; ind3 < sizes[3]; ind3++) {
        for (size_t ind4 = 0L; ind4 < sizes[4]; ind4++) {
            double* Atp = Ap 
                + (Ainds[0].first + ind0) * Astrides[0]
                + (Ainds[1].first + ind1) * Astrides[1]
                + (Ainds[2].first + ind2) * Astrides[2]
                + (Ainds[3].first + ind3) * Astrides[3]
                + (Ainds[4].first + ind4) * Astrides[4]
                + Ainds[slow_dims].first * Astrides[slow_dims]; 
            double* Ctp = Cp 
                + (Cinds[0].first + ind0) * Cstrides[0]
                + (Cinds[1].first + ind1) * Cstrides[1]
                + (Cinds[2].first + ind2) * Cstrides[2]
                + (Cinds[3].first + ind3) * Cstrides[3]
                + (Cinds[4].first + ind4) * Cstrides[4]
                + Cinds[slow_dims].first * Cstrides[slow_dims]; 
            C_DSCAL(fast_size,beta,Ctp,1);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }}}}}
    } else if (slow_dims == 6) {
        #pragma omp parallel for
        for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++) {
        for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++) {
        for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++) {
        for (size_t ind3 = 0L; ind3 < sizes[3]; ind3++) {
        for (size_t ind4 = 0L; ind4 < sizes[4]; ind4++) {
        for (size_t ind5 = 0L; ind5 < sizes[5]; ind5++) {
            double* Atp = Ap 
                + (Ainds[0].first + ind0) * Astrides[0]
                + (Ainds[1].first + ind1) * Astrides[1]
                + (Ainds[2].first + ind2) * Astrides[2]
                + (Ainds[3].first + ind3) * Astrides[3]
                + (Ainds[4].first + ind4) * Astrides[4]
                + (Ainds[5].first + ind5) * Astrides[5]
                + Ainds[slow_dims].first * Astrides[slow_dims]; 
            double* Ctp = Cp 
                + (Cinds[0].first + ind0) * Cstrides[0]
                + (Cinds[1].first + ind1) * Cstrides[1]
                + (Cinds[2].first + ind2) * Cstrides[2]
                + (Cinds[3].first + ind3) * Cstrides[3]
                + (Cinds[4].first + ind4) * Cstrides[4]
                + (Cinds[5].first + ind5) * Cstrides[5]
                + Cinds[slow_dims].first * Cstrides[slow_dims]; 
            C_DSCAL(fast_size,beta,Ctp,1);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }}}}}}
    } else if (slow_dims == 7) {
        #pragma omp parallel for
        for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++) {
        for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++) {
        for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++) {
        for (size_t ind3 = 0L; ind3 < sizes[3]; ind3++) {
        for (size_t ind4 = 0L; ind4 < sizes[4]; ind4++) {
        for (size_t ind5 = 0L; ind5 < sizes[5]; ind5++) {
        for (size_t ind6 = 0L; ind6 < sizes[6]; ind6++) {
            double* Atp = Ap 
                + (Ainds[0].first + ind0) * Astrides[0]
                + (Ainds[1].first + ind1) * Astrides[1]
                + (Ainds[2].first + ind2) * Astrides[2]
                + (Ainds[3].first + ind3) * Astrides[3]
                + (Ainds[4].first + ind4) * Astrides[4]
                + (Ainds[5].first + ind5) * Astrides[5]
                + (Ainds[6].first + ind6) * Astrides[6]
                + Ainds[slow_dims].first * Astrides[slow_dims]; 
            double* Ctp = Cp 
                + (Cinds[0].first + ind0) * Cstrides[0]
                + (Cinds[1].first + ind1) * Cstrides[1]
                + (Cinds[2].first + ind2) * Cstrides[2]
                + (Cinds[3].first + ind3) * Cstrides[3]
                + (Cinds[4].first + ind4) * Cstrides[4]
                + (Cinds[5].first + ind5) * Cstrides[5]
                + (Cinds[6].first + ind6) * Cstrides[6]
                + Cinds[slow_dims].first * Cstrides[slow_dims]; 
            C_DSCAL(fast_size,beta,Ctp,1);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }}}}}}}
    } else {
        #pragma omp parallel for
        for (size_t ind = 0L; ind < slow_size; ind++) {
            double* Ctp = Cp;
            double* Atp = Ap;
            size_t num = ind;
            for (int dim = slow_dims - 1; dim >= 0; dim--) {
                size_t val = num % sizes[dim]; // value of the dim-th index
                num /= sizes[dim];
                Atp += (Ainds[dim].first + val) * Astrides[dim];
                Ctp += (Cinds[dim].first + val) * Cstrides[dim];
            }
            Atp += Ainds[slow_dims].first * Astrides[slow_dims];
            Ctp += Cinds[slow_dims].first * Cstrides[slow_dims];
            C_DSCAL(fast_size,beta,Ctp,1);
            C_DAXPY(fast_size,alpha,Atp,1,Ctp,1);
        }
    }
}
std::map<std::string, TensorImplPtr> CoreTensorImpl::syev(EigenvalueOrder order) const
{
    squareCheck(this, true);

    CoreTensorImpl *vecs = new CoreTensorImpl("Eigenvectors of " + name(), dims());
    CoreTensorImpl *vals = new CoreTensorImpl("Eigenvalues of " + name(), {dims()[0]});

    vecs->copy(this, 1.0);

    size_t n = dims()[0];
    size_t lwork = 3 * dims()[0];
    double *work = new double[lwork];
    C_DSYEV('V', 'U', n, vecs->data_, n, vals->data_, work, lwork);

    //If descending is required, the canonical order must be reversed
    //Sort is stable
    if (order == kDescending) {
        double* Temp_sqrsp_col = memory::allocate<double>(n);
        double w_Temp_sqrsp;

        for (size_t c = 0; c<n/2; c++) {

            //Swap eigenvectors
            C_DCOPY(n, vecs->data_ + c,     n, Temp_sqrsp_col,      1);
            C_DCOPY(n, vecs->data_ + n-c-1, n, vecs->data_ + c,     n);
            C_DCOPY(n, Temp_sqrsp_col,      1, vecs->data_ + n-c-1, n);

            //Swap eigenvalues
            w_Temp_sqrsp = vals->data_[c];
            vals->data_[c] = vals->data_[n-c-1];
            vals->data_[n-c-1] = w_Temp_sqrsp;

        }

        memory::free(Temp_sqrsp_col);
    }

    std::map<std::string, TensorImplPtr> result;
    result["eigenvectors"] = vecs;
    result["eigenvalues"] = vals;

    return result;
}

std::map<std::string, TensorImplPtr> CoreTensorImpl::geev(EigenvalueOrder /*order*/) const
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

TensorImplPtr CoreTensorImpl::power(double alpha, double condition) const
{
    // this call will ensure squareness
    std::map<std::string, TensorImplPtr> diag = syev(kAscending);

    size_t n = diag["eigenvalues"]->dims()[0];
    double *a = dynamic_cast<CoreTensorImplPtr>(diag["eigenvalues"])->data();
    double *a1 = dynamic_cast<CoreTensorImplPtr>(diag["eigenvectors"])->data();
    double *a2 = memory::allocate<double>(n*n);

    memcpy(a2, a1, sizeof(double)*n*n);

    double max_a = (std::fabs(a[n-1]) > std::fabs(a[0]) ? std::fabs(a[n-1]) : std::fabs(a[0]));
    int remain = 0;
    for (size_t i=0; i<n; i++) {

        if (alpha < 0.0 && fabs(a[i]) < condition * max_a)
            a[i] = 0.0;
        else {
            a[i] = pow(a[i], alpha);
            if (std::isfinite(a[i])) {
                remain++;
            } else {
                a[i] = 0.0;
            }
        }

        C_DSCAL(n, a[i], a2 + (i*n), 1);
    }

    CoreTensorImpl *powered = new CoreTensorImpl(name() + "^" + std::to_string(alpha), dims());

    C_DGEMM('T','N',n,n,n,1.0,a2,n,a1,n,0.0,powered->data_,n);

    memory::free(a2);

    // Need to manually delete the tensors in the diag map
    for (auto& el : diag) {
        delete el.second;
    }

    return powered;
}

void CoreTensorImpl::givens(int /*dim*/, int /*i*/, int /*j*/, double /*s*/, double /*c*/)
{
    ThrowNotImplementedException;
}

}
