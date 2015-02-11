#if !defined(TENSOR_CYCLOPS_H)
#define TENSOR_CYCLOPS_H

#include "tensor/tensorimpl.h"
#include <ctf.hpp>
#include <El.hpp>

namespace tensor {

namespace cyclops {

int initialize(int argc, char* argv[]);
void finalize();

class CyclopsTensorImpl : public TensorImpl
{
public:
    CyclopsTensorImpl(const std::string& name, const Dimension& dims);
    ~CyclopsTensorImpl();

    CTF_Tensor* cyclops() const { return cyclops_; }

    // => Simple Single Tensor Operations <= //

    double norm(
            int type = 2) const;

    void scale(double beta = 0.0);

    void permute(
            ConstTensorImplPtr A,
            const std::vector<std::string>& Cinds,
            const std::vector<std::string>& Ainds,
            double alpha = 1.0,
            double beta = 0.0);

    void contract(
            ConstTensorImplPtr A,
            ConstTensorImplPtr B,
            const std::vector<std::string>& Cinds,
            const std::vector<std::string>& Ainds,
            const std::vector<std::string>& Binds,
            double alpha = 1.0,
            double beta = 0.0);

    // => Order-2 Operations <= //

    std::map<std::string, TensorImplPtr> syev(EigenvalueOrder order) const;

private:

#if defined(HAVE_ELEMENTAL)
    // => Order-2 Helper Functions <=
    void copyToElemental2(El::DistMatrix<double>& x) const;
    void copyFromElemental2(const El::DistMatrix<double>& x);

    // => Order-1 Helper Functions <=
    void copyFromElemental1(const El::DistMatrix<double, El::VR, El::STAR>& x);
#endif

    CTF_Tensor *cyclops_;
};

}

typedef cyclops::CyclopsTensorImpl* CyclopsTensorImplPtr;
typedef const cyclops::CyclopsTensorImpl* ConstCyclopsTensorImplPtr;

}

#endif
