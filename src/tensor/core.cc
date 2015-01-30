#include "tensorimpl.h"
#include "core.h"
#include "memory.h"
#include "math/math.h"
#include "indices.h"
#include <string.h>

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

double CoreTensorImpl::norm(double power) const
{
    ThrowNotImplementedException;
}
double CoreTensorImpl::rms(double power) const
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

    return C_DDOT(numel(), data_, 1, ((ConstCoreTensorImplPtr)x)->data(), 1);
}

void CoreTensorImpl::contract(
    ConstTensorImplPtr A,
    ConstTensorImplPtr B,
    const std::vector<std::string>& Cinds,
    const std::vector<std::string>& Ainds,
    const std::vector<std::string>& Binds,
    double alpha,
    double beta)
{
    CoreContractionManager manager(*this,*(const CoreTensorImplPtr)A,*(const CoreTensorImplPtr)B,Cinds,Ainds,Binds,alpha,beta);
    manager.contract();
}

void CoreTensorImpl::contract(ConstTensorImplPtr A, ConstTensorImplPtr B, const ContractionTopology &topology,
                              double alpha, double beta)
{

    CoreTensorContractionTopology manager(topology,*this,*(const CoreTensorImplPtr)A,*(const CoreTensorImplPtr)B);
    manager.contract(alpha,beta);
}
void CoreTensorImpl::permute(
    ConstTensorImplPtr A, 
    const std::vector<std::string>& CindsS,
    const std::vector<std::string>& AindsS)
{
    // => Convert to indices of A <= //

    std::vector<int> Ainds = indices::permutation_order(CindsS, AindsS);
    for (int dim = 0; dim < rank(); dim++) {
        if (dims()[dim] != A->dims()[Ainds[dim]])
            throw std::runtime_error("Permuted tensors do not have same dimensions");
    }

    /// Data pointers
    double* Cp = data();
    double* Ap = ((const CoreTensorImplPtr)A)->data();

    // => Index Logic <= //

    /// Determine the number of united fast indices and memcpy size
    /// C_ij = A_ji would have no fast dimensions and a fast size of 1
    /// C_ijk = A_jik would have k as a fast dimension, and a fast size of dim(k)
    /// C_ijkl = A_jikl would have k and l as fast dimensions, and a fast size of dim(k) * dim(l)
    int fast_dims = 0;
    size_t fast_size = 1L;
    for (int dim = ((int)rank()) - 1; dim >= 0; dim--) {
        if (dim == Ainds[dim]) {
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
        ::memcpy(Cp,Ap,sizeof(double)*fast_size);
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
    
    // => Actual Permute Operation <= //

    if (slow_dims == 2) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0]; 
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
            double* Atp = Ap + 
                Cind0 * AstridesC[0] +
                Cind1 * AstridesC[1];
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
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
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
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
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
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
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
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
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
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
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
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
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
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
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
        }
    }
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

CoreTensorContractionTopology::CoreTensorContractionTopology(
    const ContractionTopology& topology,
    const CoreTensorImpl& C,
    const CoreTensorImpl& A,
    const CoreTensorImpl& B) :
    topology_(topology),
    C_(C),
    A_(A),
    B_(B)
{
    std::vector<std::pair<int, std::string> > PC;
    std::vector<std::pair<int, std::string> > PA;
    std::vector<std::pair<int, std::string> > PB;
    std::vector<std::pair<int, std::string> > iA;
    std::vector<std::pair<int, std::string> > iC;
    std::vector<std::pair<int, std::string> > jB;
    std::vector<std::pair<int, std::string> > jC;
    std::vector<std::pair<int, std::string> > kA;
    std::vector<std::pair<int, std::string> > kB;

    const std::vector<std::string>& indices = topology_.indices();
    const std::vector<int>& A_pos = topology_.A_pos();
    const std::vector<int>& B_pos = topology_.B_pos();
    const std::vector<int>& C_pos = topology_.C_pos();
    const std::vector<ContractionType>& types = topology_.types();

    for (size_t ind = 0L; ind < indices.size(); ind++) {
        ContractionType type = types[ind];
        std::string tag = indices[ind];
        if (type == ABC) {
            PC.push_back(std::pair<int, std::string>(C_pos[ind],tag));
            PA.push_back(std::pair<int, std::string>(A_pos[ind],tag));
            PB.push_back(std::pair<int, std::string>(B_pos[ind],tag));
        } else if (type == AC) {
            iC.push_back(std::pair<int, std::string>(C_pos[ind],tag));
            iA.push_back(std::pair<int, std::string>(A_pos[ind],tag));
        } else if (type == BC) {
            jC.push_back(std::pair<int, std::string>(C_pos[ind],tag));
            jB.push_back(std::pair<int, std::string>(B_pos[ind],tag));
        } else if (type == AB) {
            kA.push_back(std::pair<int, std::string>(A_pos[ind],tag));
            kB.push_back(std::pair<int, std::string>(B_pos[ind],tag));
        }
    }

    std::sort(PC.begin(),PC.end());
    std::sort(PA.begin(),PA.end());
    std::sort(PB.begin(),PB.end());
    std::sort(iC.begin(),iC.end());
    std::sort(iA.begin(),iA.end());
    std::sort(jC.begin(),jC.end());
    std::sort(jB.begin(),jB.end());
    std::sort(kA.begin(),kA.end());
    std::sort(kB.begin(),kB.end());

    std::vector<std::string> compound_names;
    std::vector<std::vector<std::pair<int, std::string> > > compound_inds;

    compound_names.push_back("PC");
    compound_names.push_back("PA");
    compound_names.push_back("PB");
    compound_names.push_back("iC");
    compound_names.push_back("iA");
    compound_names.push_back("jC");
    compound_names.push_back("jB");
    compound_names.push_back("kA");
    compound_names.push_back("kB");

    compound_inds.push_back(PC);
    compound_inds.push_back(PA);
    compound_inds.push_back(PB);
    compound_inds.push_back(iC);
    compound_inds.push_back(iA);
    compound_inds.push_back(jC);
    compound_inds.push_back(jB);
    compound_inds.push_back(kA);
    compound_inds.push_back(kB);

    // Contiguous Index Test
    for (size_t ind = 0L; ind < compound_names.size(); ind++) {
        const std::vector<std::pair<int, std::string> >& compound_ind = compound_inds[ind];
        for (int prim = 0L; prim < ((int)compound_ind.size()) - 1; prim++) {
            if (compound_ind[prim+1].first != compound_ind[prim].first + 1) {
                throw std::runtime_error("Index is not contiguous:" + compound_names[ind]);
            }
        }
    }

    // Permutation Test
    for (size_t prim = 0L; prim < PC.size(); prim++) {
        if (PC[prim].second != PA[prim].second || PC[prim].second != PB[prim].second) {
            throw std::runtime_error("P indices are not all in same permutation");
        }
    }
    for (size_t prim = 0L; prim < iC.size(); prim++) {
        if (iC[prim].second != iA[prim].second) {
            throw std::runtime_error("i indices are not all in same permutation");
        }
    }
    for (size_t prim = 0L; prim < jC.size(); prim++) {
        if (jC[prim].second != jB[prim].second) {
            throw std::runtime_error("j indices are not all in same permutation");
        }
    }
    for (size_t prim = 0L; prim < kA.size(); prim++) {
        if (kA[prim].second != kB[prim].second) {
            throw std::runtime_error("k indices are not all in same permutation");
        }
    }

    // Hadamard Test
    int Psize = PC.size();
    if (Psize) {
        if (PC[0].first != 0) throw std::runtime_error("PC is not first index");
        if (PA[0].first != 0) throw std::runtime_error("PA is not first index");
        if (PB[0].first != 0) throw std::runtime_error("PB is not first index");
    }

    C_transpose_ = false;
    A_transpose_ = false;
    B_transpose_ = false;
    if (iC.size() && iC[0].first != Psize) C_transpose_ = true;
    if (iA.size() && iA[0].first != Psize) A_transpose_ = true;
    if (jB.size() && jB[0].first == Psize) B_transpose_ = true;

    ABC_size_ = 1L;
    AC_size_ = 1L;
    BC_size_ = 1L;
    AB_size_ = 1L;
    for (size_t prim = 0L; prim < PC.size(); prim++) {
        size_t size1 = C.dims()[PC[prim].first];
        size_t size2 = A.dims()[PA[prim].first];
        size_t size3 = B.dims()[PB[prim].first];
        if (size1 != size2 || size1 != size3)
            throw std::runtime_error("Hadamard indices are not same size");
        ABC_size_ *= size1;
    }
    for (size_t prim = 0L; prim < iC.size(); prim++) {
        size_t size1 = C.dims()[iC[prim].first];
        size_t size2 = A.dims()[iA[prim].first];
        if (size1 != size2)
            throw std::runtime_error("i indices are not same size");
        AC_size_ *= size1;
    }
    for (size_t prim = 0L; prim < jC.size(); prim++) {
        size_t size1 = C.dims()[jC[prim].first];
        size_t size2 = B.dims()[jB[prim].first];
        if (size1 != size2)
            throw std::runtime_error("j indices are not same size");
        BC_size_ *= size1;
    }
    for (size_t prim = 0L; prim < kA.size(); prim++) {
        size_t size1 = A.dims()[kA[prim].first];
        size_t size2 = B.dims()[kB[prim].first];
        if (size1 != size2)
            throw std::runtime_error("k indices are not same size");
        AB_size_ *= size1;
    }

}

void CoreTensorContractionTopology::contract(double alpha, double beta)
{
    double* Ap = A_.data();
    double* Bp = B_.data();
    double* Cp = C_.data();
    for (size_t P = 0L; P < ABC_size_; P++) {

        char transL;
        char transR;
        size_t nrow;
        size_t ncol;
        double* Lp;
        double* Rp;
        size_t ldaL;
        size_t ldaR;

        if (C_transpose_) {
            Lp = Bp;
            Rp = Ap;
            nrow = BC_size_;
            ncol = AC_size_;
            transL = (B_transpose_ ? 'N' : 'T');
            transR = (A_transpose_ ? 'N' : 'T');
            ldaL = (B_transpose_ ? AB_size_ : BC_size_);
            ldaR = (A_transpose_ ? AC_size_ : AB_size_);
        } else {
            Lp = Ap;
            Rp = Bp;
            nrow = AC_size_;
            ncol = BC_size_;
            transL = (A_transpose_ ? 'T' : 'N');
            transR = (B_transpose_ ? 'T' : 'N');
            ldaL = (A_transpose_ ? AC_size_ : AB_size_);
            ldaR = (B_transpose_ ? AB_size_ : BC_size_);
        }

        size_t nzip = AB_size_;
        size_t ldaC = (C_transpose_ ? AC_size_ : BC_size_);

        C_DGEMM(transL,transR,nrow,ncol,nzip,alpha,Lp,ldaL,Rp,ldaR,beta,Cp,ldaC);

        Cp += AC_size_ * BC_size_;
        Ap += AB_size_ * AC_size_;
        Bp += AB_size_ * BC_size_;
    }
}

namespace indices {

int find_index_in_vector(const std::vector<std::string>& vec, const std::string& key)
{
    for (size_t ind = 0L; ind < vec.size(); ind++) {
        if (key == vec[ind]) {
            return ind;
        }
    } 
    return -1;
}
bool is_contiguous(const std::vector<std::pair<int, std::string>>& vec) 
{
    for (int prim = 0L; prim < ((int)vec.size()) - 1; prim++) {
        if (vec[prim+1].first != vec[prim].first + 1) {
            return false;
        }
    }
    return true;
}
bool is_equivalent(const std::vector<std::string>& vec1, const std::vector<std::string>& vec2) 
{
    for (int prim = 0L; prim < vec1.size(); prim++) {
        if (vec1[prim] != vec2[prim]) {
            return false;
        }
    }
    return true;
}
Dimension permuted_dimension(
    const Dimension& old_dim, 
    const std::vector<std::string>& new_order,
    const std::vector<std::string>& old_order)
{
    std::vector<int> order = indices::permutation_order(new_order,old_order);
    Dimension new_dim(order.size(),0L);
    for (size_t ind = 0L; ind < order.size(); ind++) {
        new_dim[ind] = old_dim[order[ind]];
    }
    return new_dim;
}

}

void CoreContractionManager::contract()
{
    // => Permutation Logic <= //

    // Determine unique indices
    std::vector<std::string> inds;
    inds.insert(inds.end(),Cinds_.begin(),Cinds_.end());
    inds.insert(inds.end(),Ainds_.begin(),Ainds_.end());
    inds.insert(inds.end(),Binds_.begin(),Binds_.end());
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
        int Cpos = indices::find_index_in_vector(Cinds_,index);
        int Apos = indices::find_index_in_vector(Ainds_,index);
        int Bpos = indices::find_index_in_vector(Binds_,index);
        if (Cpos != -1 && Apos != -1 && Bpos != -1) {
            if (C_.dims()[Cpos] != A_.dims()[Apos] || C_.dims()[Cpos] != B_.dims()[Bpos]) 
                throw std::runtime_error("Invalid ABC (Hadamard) index size");
            compound_inds["PC"].push_back(std::make_pair(Cpos,index));
            compound_inds["PA"].push_back(std::make_pair(Apos,index));
            compound_inds["PB"].push_back(std::make_pair(Bpos,index));
            ABC_size *= C_.dims()[Cpos];
        } else if (Cpos != -1 && Apos != -1 && Bpos == -1) {
            if (C_.dims()[Cpos] != A_.dims()[Apos])
                throw std::runtime_error("Invalid AC (Left) index size");
            compound_inds["iC"].push_back(std::make_pair(Cpos,index));
            compound_inds["iA"].push_back(std::make_pair(Apos,index));
            AC_size *= C_.dims()[Cpos];
        } else if (Cpos != -1 && Apos == -1 && Bpos != -1) {
            if (C_.dims()[Cpos] != B_.dims()[Bpos])
                throw std::runtime_error("Invalid BC (Right) index size");
            compound_inds["jC"].push_back(std::make_pair(Cpos,index));
            compound_inds["jB"].push_back(std::make_pair(Bpos,index));
            BC_size *= C_.dims()[Cpos];
        } else if (Cpos == -1 && Apos != -1 && Bpos != -1) {
            if (A_.dims()[Apos] != B_.dims()[Bpos])
                throw std::runtime_error("Invalid AB (Contraction) index size");
            compound_inds["kA"].push_back(std::make_pair(Apos,index));
            compound_inds["kB"].push_back(std::make_pair(Bpos,index));
            AB_size *= B_.dims()[Bpos];
        } else {
            throw std::runtime_error("Invalid contraction topology - index only occurs once.");
        }
    }

    // Sort compound indices by primitive indices to determine continuity
    for (size_t ind = 0L; ind < compound_names.size(); ind++) {
        std::sort(compound_inds[compound_names[ind]].begin(), compound_inds[compound_names[ind]].end());
    } 

    // The list to mark for permutation [C,A,B]
    std::vector<bool> perms(3,false);

    // Contiguous Index Test (always requires permutation)
    perms[0] = perms[0] || !indices::is_contiguous(compound_inds["PC"]);
    perms[0] = perms[0] || !indices::is_contiguous(compound_inds["iC"]);
    perms[0] = perms[0] || !indices::is_contiguous(compound_inds["jC"]);
    perms[1] = perms[1] || !indices::is_contiguous(compound_inds["PA"]);
    perms[1] = perms[1] || !indices::is_contiguous(compound_inds["iA"]);
    perms[1] = perms[1] || !indices::is_contiguous(compound_inds["kA"]);
    perms[2] = perms[2] || !indices::is_contiguous(compound_inds["PB"]);
    perms[2] = perms[2] || !indices::is_contiguous(compound_inds["jB"]);
    perms[2] = perms[2] || !indices::is_contiguous(compound_inds["kB"]);

    // Hadamard Test (always requires permutation)
    int Psize = compound_inds["PC"].size();
    if (Psize) {
        perms[0] = perms[0] || (compound_inds["PC"][0].first != 0);
        perms[1] = perms[1] || (compound_inds["PA"][0].first != 0);
        perms[2] = perms[2] || (compound_inds["PB"][0].first != 0);
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
    if (!indices::is_equivalent(compound_inds2["iC"],compound_inds2["iA"])) {
        if (perms[0]) {
            compound_inds2["iC"] = compound_inds2["iA"];
        } else if (perms[1]) {
            compound_inds2["iA"] = compound_inds2["iC"];
        } else if (C_.numel() <= A_.numel()) {
            compound_inds2["iC"] = compound_inds2["iA"];
            perms[0] = true; 
        } else {
            compound_inds2["iA"] = compound_inds2["iC"];
            perms[1] = true; 
        }
    }
    if (!indices::is_equivalent(compound_inds2["jC"],compound_inds2["jB"])) {
        if (perms[0]) {
            compound_inds2["jC"] = compound_inds2["jB"];
        } else if (perms[2]) {
            compound_inds2["jB"] = compound_inds2["jC"];
        } else if (C_.numel() <= B_.numel()) {
            compound_inds2["jC"] = compound_inds2["jB"];
            perms[0] = true; 
        } else {
            compound_inds2["jB"] = compound_inds2["jC"];
            perms[2] = true; 
        }
    }
    if (!indices::is_equivalent(compound_inds2["kA"],compound_inds2["kB"])) {
        if (perms[1]) {
            compound_inds2["kA"] = compound_inds2["kB"];
        } else if (perms[2]) {
            compound_inds2["kB"] = compound_inds2["kA"];
        } else if (A_.numel() <= B_.numel()) {
            compound_inds2["kA"] = compound_inds2["kB"];
            perms[1] = true; 
        } else {
            compound_inds2["kB"] = compound_inds2["kA"];
            perms[2] = true; 
        }
    }
    if (!indices::is_equivalent(compound_inds2["PC"],compound_inds2["PA"])) {
        compound_inds2["PA"] = compound_inds2["PC"];
        perms[1] = true; 
    }
    if (!indices::is_equivalent(compound_inds2["PC"],compound_inds2["PB"])) {
        compound_inds2["PB"] = compound_inds2["PC"];
        perms[2] = true; 
    }

    /// Assign the permuted indices (if flagged for permute) or the original indices
    std::vector<std::string> Cinds2;
    std::vector<std::string> Ainds2;
    std::vector<std::string> Binds2;
    if (perms[0]) {
        Cinds2.insert(Cinds2.end(),compound_inds2["PC"].begin(),compound_inds2["PC"].end());
        Cinds2.insert(Cinds2.end(),compound_inds2["iC"].begin(),compound_inds2["iC"].end());
        Cinds2.insert(Cinds2.end(),compound_inds2["jC"].begin(),compound_inds2["jC"].end());
        C_transpose = false;
    } else {
        Cinds2 = Cinds_;
    }
    if (perms[1]) {
        Ainds2.insert(Ainds2.end(),compound_inds2["PA"].begin(),compound_inds2["PA"].end());
        Ainds2.insert(Ainds2.end(),compound_inds2["iA"].begin(),compound_inds2["iA"].end());
        Ainds2.insert(Ainds2.end(),compound_inds2["kA"].begin(),compound_inds2["kA"].end());
        A_transpose = false;
    } else {
        Ainds2 = Ainds_;
    }
    if (perms[2]) {
        Binds2.insert(Binds2.end(),compound_inds2["PB"].begin(),compound_inds2["PB"].end());
        Binds2.insert(Binds2.end(),compound_inds2["jB"].begin(),compound_inds2["jB"].end());
        Binds2.insert(Binds2.end(),compound_inds2["kB"].begin(),compound_inds2["kB"].end());
        B_transpose = true;
    } else {
        Binds2 = Binds_;
    }

    // So what exactly happened?
    /**
    printf("==> Core Contraction <==\n\n");
    printf("Original: C[");
    for (size_t ind = 0l; ind < Cinds_.size(); ind++) {
        printf("%s", Cinds_[ind].c_str());
    }
    printf("] = A["); 
    for (size_t ind = 0l; ind < Ainds_.size(); ind++) {
        printf("%s", Ainds_[ind].c_str());
    }
    printf("] * B["); 
    for (size_t ind = 0l; ind < Binds_.size(); ind++) {
        printf("%s", Binds_[ind].c_str());
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
    printf("C Permuted: %s\n", perms[0] ? "Yes" : "No");
    printf("A Permuted: %s\n", perms[1] ? "Yes" : "No");
    printf("B Permuted: %s\n", perms[2] ? "Yes" : "No");
    printf("\n");
    **/

    // => Alias or Allocate A, B, C <= //

    Dimension Cdims2 = indices::permuted_dimension(C_.dims(), Cinds2, Cinds_);
    Dimension Adims2 = indices::permuted_dimension(A_.dims(), Ainds2, Ainds_);
    Dimension Bdims2 = indices::permuted_dimension(B_.dims(), Binds2, Binds_);

    double* Cp = C_.data();
    double* Ap = A_.data();
    double* Bp = B_.data();
    double* C2p = C_.data();
    double* A2p = A_.data();
    double* B2p = B_.data();

    /// TODO: This is ugly. Overall, where do we use shared pointers, references, const references, or object copy?
    boost::shared_ptr<CoreTensorImpl> C2;
    boost::shared_ptr<CoreTensorImpl> B2;
    boost::shared_ptr<CoreTensorImpl> A2;
    if (perms[0]) {
        C2 = boost::shared_ptr<CoreTensorImpl>(new CoreTensorImpl("C2", Cdims2));
        C2p = C2->data();
    }
    if (perms[1]) {
        A2 = boost::shared_ptr<CoreTensorImpl>(new CoreTensorImpl("A2", Adims2));
        A2p = A2->data();
    }
    if (perms[2]) {
        B2 = boost::shared_ptr<CoreTensorImpl>(new CoreTensorImpl("B2", Bdims2));
        B2p = B2->data();
    }
    
    // => Permute A, B, and C if Necessary <= //

    if (perms[0]) C2->permute(&C_,Cinds2,Cinds_);
    if (perms[1]) A2->permute(&A_,Ainds2,Ainds_);
    if (perms[2]) B2->permute(&B_,Binds2,Binds_);

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

        C_DGEMM(transL,transR,nrow,ncol,nzip,alpha_,Lp,ldaL,Rp,ldaR,beta_,C2p,ldaC);

        C2p += AC_size * BC_size;
        A2p += AB_size * AC_size;
        B2p += AB_size * BC_size;
    }

    // => Permute C if Necessary <= //
    
    if (perms[0]) C_.permute(C2.get(),Cinds_,Cinds2);

}

}
