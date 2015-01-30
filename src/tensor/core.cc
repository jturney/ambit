#include "tensorimpl.h"
#include "core.h"
#include "memory.h"
#include "math/math.h"
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

void CoreTensorImpl::contract(ConstTensorImplPtr A, ConstTensorImplPtr B, const ContractionTopology &topology,
                              double alpha, double beta)
{

    CoreTensorContractionTopology manager(topology,*this,*(const CoreTensorImplPtr)A,*(const CoreTensorImplPtr)B);
    manager.contract(alpha,beta);
}
void CoreTensorImpl::permute(ConstTensorImplPtr A, const std::vector<int>& Ainds)
{
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

    // Fully sorted case or (equivalently) 0-rank tensors
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
            double* Atp = Ap + Cind0 * AstridesC[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            Atp += AstridesC[1];
            Ctp += fast_size;
        }}
    } else if (slow_dims == 3) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0]; 
            double* Atp = Ap + Cind0 * AstridesC[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            Ctp += fast_size;
            Atp += AstridesC[2];
        }
            Atp += AstridesC[1];
        }}
    } else if (slow_dims == 4) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0]; 
            double* Atp = Ap + Cind0 * AstridesC[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            Ctp += fast_size;
            Atp += AstridesC[3];
        }
            Atp += AstridesC[2];
        }
            Atp += AstridesC[1];
        }}
    } else if (slow_dims == 5) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0]; 
            double* Atp = Ap + Cind0 * AstridesC[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
        for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++) {
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            Ctp += fast_size;
            Atp += AstridesC[4];
        }
            Atp += AstridesC[3];
        }
            Atp += AstridesC[2];
        }
            Atp += AstridesC[1];
        }}
    } else if (slow_dims == 6) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0]; 
            double* Atp = Ap + Cind0 * AstridesC[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
        for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++) {
        for (size_t Cind5 = 0L; Cind5 < Csizes[5]; Cind5++) {
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            Ctp += fast_size;
            Atp += AstridesC[5];
        }
            Atp += AstridesC[4];
        }
            Atp += AstridesC[3];
        }
            Atp += AstridesC[2];
        }
            Atp += AstridesC[1];
        }}
    } else if (slow_dims == 7) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0]; 
            double* Atp = Ap + Cind0 * AstridesC[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
        for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++) {
        for (size_t Cind5 = 0L; Cind5 < Csizes[5]; Cind5++) {
        for (size_t Cind6 = 0L; Cind6 < Csizes[6]; Cind6++) {
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            Ctp += fast_size;
            Atp += AstridesC[6];
        }
            Atp += AstridesC[5];
        }
            Atp += AstridesC[4];
        }
            Atp += AstridesC[3];
        }
            Atp += AstridesC[2];
        }
            Atp += AstridesC[1];
        }}
    } else if (slow_dims == 8) {
        #pragma omp parallel for
        for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++) {
            double* Ctp = Cp + Cind0 * Cstrides[0]; 
            double* Atp = Ap + Cind0 * AstridesC[0];
        for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++) {
        for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++) {
        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++) {
        for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++) {
        for (size_t Cind5 = 0L; Cind5 < Csizes[5]; Cind5++) {
        for (size_t Cind6 = 0L; Cind6 < Csizes[6]; Cind6++) {
        for (size_t Cind7 = 0L; Cind7 < Csizes[7]; Cind7++) {
            ::memcpy(Ctp,Atp,sizeof(double)*fast_size);
            Ctp += fast_size;
            Atp += AstridesC[7];
        }
            Atp += AstridesC[6];
        }
            Atp += AstridesC[5];
        }
            Atp += AstridesC[4];
        }
            Atp += AstridesC[3];
        }
            Atp += AstridesC[2];
        }
            Atp += AstridesC[1];
        }}
    } else {
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
    squareCheck(this, true);

    CoreTensorImpl *vecs = new CoreTensorImpl("Eigenvectors of " + name(), dims());
    CoreTensorImpl *vals = new CoreTensorImpl("Eigenvalues of " + name(), {dims()[0]});

    vecs->copy(this, 1.0);

    size_t n = dims()[0];
    size_t lwork = 3 * dims()[0];
    double *work = new double[lwork];
    C_DSYEV('V', 'U', n, vecs->data_, n, vals->data_, work, lwork);

    std::map<std::string, TensorImplPtr> result;
    result["eigenvectors"] = vecs;
    result["eigenvalues"] = vals;

    return result;
}

std::map<std::string, TensorImplPtr> CoreTensorImpl::geev(EigenvalueOrder order) const
{
//    squareCheck(this, true);
//
//    CoreTensorImpl *L = new CoreTensorImpl("Left eigenvectors of " + name(), dims());
//    CoreTensorImpl *R = new CoreTensorImpl("Right eigenvectors of " + name(), dims());
//    CoreTensorImpl *work = new CoreTensorImpl("Work of " + name(), dims());
//    CoreTensorImpl *vals = new CoreTensorImpl("Eigenvalues of " + name(), {dims()[0]});
//
//    work->copy(this, 1.0);
//
//    size_t n = dims()[0];
//    size_t work = 4*n;
//    double *
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

}
