#if !defined(TENSOR_CORE_H)
#define TENSOR_CORE_H

#include "tensor/tensorimpl.h"

namespace ambit {

class CoreTensorImpl : public TensorImpl
{
public:
    CoreTensorImpl(const std::string& name, const Dimension& dims);

    std::vector<double>& data() { return data_; }
    const std::vector<double>& data() const { return data_; }

    // => Simple Single Tensor Operations <= //

    double norm(
        int type = 2) const;

    std::tuple<double, std::vector<size_t>> max() const;

    std::tuple<double, std::vector<size_t>> min() const;

    void scale(
        double beta = 0.0);

    void set(double alpha);

    void permute(
        ConstTensorImplPtr A,
        const Indices& Cinds,
        const Indices& Ainds,
        double alpha = 1.0,
        double beta = 0.0);

    void contract(
        ConstTensorImplPtr A,
        ConstTensorImplPtr B,
        const Indices& Cinds,
        const Indices& Ainds,
        const Indices& Binds,
        double alpha = 1.0,
        double beta = 0.0);

    void gemm(
        ConstTensorImplPtr A,
        ConstTensorImplPtr B,
        bool transA,
        bool transB,
        size_t nrow,
        size_t ncol,
        size_t nzip,
        size_t ldaA,
        size_t ldaB,
        size_t ldaC,
        size_t offA = 0L,
        size_t offB = 0L,
        size_t offC = 0L,
        double alpha = 1.0,
        double beta = 0.0);

    // => Order-2 Operations <= //

    std::map<std::string, TensorImplPtr> syev(EigenvalueOrder order) const;
    std::map<std::string, TensorImplPtr> geev(EigenvalueOrder order) const;
    //std::map<std::string, TensorImplPtr> svd() const;

    //TensorImplPtr cholesky() const;
    //std::map<std::string, TensorImplPtr> lu() const;
    //std::map<std::string, TensorImplPtr> qr() const;

    //TensorImplPtr cholesky_inverse() const;
    //TensorImplPtr inverse() const;
    TensorImplPtr power(double power, double condition = 1.0E-12) const;

    void iterate(const std::function<void (const std::vector<size_t>&, double&)>& func);
    void citerate(const std::function<void (const std::vector<size_t>&, const double&)>& func) const;

private:
    std::vector<double> data_;
};

typedef CoreTensorImpl* CoreTensorImplPtr;
typedef const CoreTensorImpl* ConstCoreTensorImplPtr;

}

#endif
