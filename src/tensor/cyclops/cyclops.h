#if !defined(TENSOR_CYCLOPS_H)
#define TENSOR_CYCLOPS_H

#include "tensor/tensorimpl.h"
#include <ctf.hpp>
#include <El.hpp>

namespace tensor {

namespace cyclops {

int initialize(int argc, char* argv[]);

class CyclopsTensorImpl : public TensorImpl
{
public:
    CyclopsTensorImpl(const std::string& name, const Dimension& dims);
    ~CyclopsTensorImpl();

    void set_data(double* data, const IndexRange& ranges = IndexRange());
    void get_data(double* data, const IndexRange& ranges = IndexRange()) const;

    // => Simple Single Tensor Operations <= //

    void zero();
    void scale(double a);
    double norm(double power = 2.0) const;
    double rms(double power = 2.0) const;

    // => Simple Double TensorImpl Operations <= //

    void scale_and_add(double a, ConstTensorImplPtr x);
    void pointwise_multiplication(ConstTensorImplPtr x);
    void pointwise_division(ConstTensorImplPtr x);
    double dot(ConstTensorImplPtr x) const;

    // => Contraction Type Operations <= //

    void contract(
                  ConstTensorImplPtr A,
                  ConstTensorImplPtr B,
                  const ContractionTopology& topology,
                  double alpha = 1.0,
                  double beta = 0.0
                  );

    // => Order-2 Operations <= //

    std::map<std::string, TensorImplPtr> syev(EigenvalueOrder order) const;
    std::map<std::string, TensorImplPtr> geev(EigenvalueOrder order) const;
    std::map<std::string, TensorImplPtr> svd() const;

    TensorImplPtr cholesky() const;
    std::map<std::string, TensorImplPtr> lu() const;
    std::map<std::string, TensorImplPtr> qr() const;

    TensorImplPtr cholesky_inverse() const;
    TensorImplPtr inverse() const;
    TensorImplPtr power(double power, double condition = 1.0E-12) const;

    void givens(int dim, int i, int j, double s, double c);

private:

#if defined(HAVE_ELEMENTAL)
    // => Order-2 Helper Functions <=
    void copyToElemental2(El::DistMatrix<double>& x) const;
    void copyFromElemental2(const El::DistMatrix<double>& x);

    // => Order-1 Helper Functions <=
    void copyFromElemental1(const El::DistMatrix<double, El::VR, El::STAR>& x);
#endif

    CTF_Tensor *data_;
};

}
}

#endif
