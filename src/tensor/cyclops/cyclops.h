#if !defined(TENSOR_CYCLOPS_H)
#define TENSOR_CYCLOPS_H

#include "tensor/tensorimpl.h"
#include <ctf.hpp>
#include <El.hpp>

namespace ambit
{

namespace cyclops
{

int initialize(int argc, char **argv);

int initialize(MPI_Comm comm, int argc = 0, char **argv = nullptr);

void finalize();

class CyclopsTensorImpl : public TensorImpl
{
  public:
    CyclopsTensorImpl(const std::string &name, const Dimension &dims);
    ~CyclopsTensorImpl();

    CTF_Tensor *cyclops() const { return cyclops_; }

    // => Simple Single Tensor Operations <= //

    double norm(int type = 2) const;

    std::tuple<double, std::vector<size_t>> max() const;

    std::tuple<double, std::vector<size_t>> min() const;

    void scale(double beta = 0.0);

    void set(double alpha);

    void permute(ConstTensorImplPtr A, const std::vector<std::string> &Cinds,
                 const std::vector<std::string> &Ainds, double alpha = 1.0,
                 double beta = 0.0);

    void contract(ConstTensorImplPtr A, ConstTensorImplPtr B,
                  const std::vector<std::string> &Cinds,
                  const std::vector<std::string> &Ainds,
                  const std::vector<std::string> &Binds, double alpha = 1.0,
                  double beta = 0.0);

    // => Order-2 Operations <= //

    std::map<std::string, TensorImplPtr> syev(EigenvalueOrder order) const;

    TensorImplPtr power(double alpha, double condition) const;

    void iterate(
        const std::function<void(const std::vector<size_t> &, double &)> &func);
    void citerate(const std::function<void(const std::vector<size_t> &,
                                           const double &)> &func) const;

  private:
#if defined(HAVE_ELEMENTAL)
    // => Order-2 Helper Functions <=
    void copyToElemental2(El::DistMatrix<double> &x) const;
    void copyFromElemental2(const El::DistMatrix<double> &x);
    void copyFromLowerElementalToFull2(const El::DistMatrix<double> &x);

    // => Order-1 Helper Functions <=
    void copyFromElemental1(const El::DistMatrix<double, El::VR, El::STAR> &x);
#endif

    CTF_Tensor *cyclops_;
};
}

typedef cyclops::CyclopsTensorImpl *CyclopsTensorImplPtr;
typedef const cyclops::CyclopsTensorImpl *ConstCyclopsTensorImplPtr;
}

#endif
