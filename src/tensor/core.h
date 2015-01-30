#if !defined(TENSOR_CORE_H)
#define TENSOR_CORE_H

#include "tensorimpl.h"

namespace tensor {

class CoreTensorImpl : public TensorImpl
{
public:
    CoreTensorImpl(const std::string& name, const Dimension& dims);
    ~CoreTensorImpl();

    void set_data(double* data, const IndexRange& ranges = IndexRange());
    void get_data(double* data, const IndexRange& ranges = IndexRange()) const;

    double* data() const { return data_; }

    // => Simple Single Tensor Operations <= //

    void zero();
    void scale(const double& a);
    double norm(double power = 2.0) const;
    double rms(double power = 2.0) const;

    // => Simple Double TensorImpl Operations <= //

    void scale_and_add(const double& a, ConstTensorImplPtr x);
    void pointwise_multiplication(ConstTensorImplPtr x);
    void pointwise_division(ConstTensorImplPtr x);
    double dot(ConstTensorImplPtr x) const;

    // => Contraction Type Operations <= //

    void contract(
        ConstTensorImplPtr A,
        ConstTensorImplPtr B,
        const std::vector<std::string>& Cinds,
        const std::vector<std::string>& Ainds,
        const std::vector<std::string>& Binds,
        double alpha = 1.0,
        double beta = 0.0);

    void permute(
        ConstTensorImplPtr A,
        const std::vector<std::string>& Cinds,
        const std::vector<std::string>& Ainds);

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
    double* data_;
};

typedef CoreTensorImpl* CoreTensorImplPtr;
typedef const CoreTensorImpl* ConstCoreTensorImplPtr;

class CoreContractionManager {

public:
    CoreContractionManager(
        CoreTensorImpl& C,
        const CoreTensorImpl& A,
        const CoreTensorImpl& B,
        const std::vector<std::string>& Cinds,
        const std::vector<std::string>& Ainds,
        const std::vector<std::string>& Binds,
        double alpha,
        double beta) :
        C_(C),
        A_(A),
        B_(B),
        Cinds_(Cinds),
        Ainds_(Ainds),
        Binds_(Binds),
        alpha_(alpha),
        beta_(beta)
    {}

    void contract();

private:    
   
    CoreTensorImpl& C_; 
    const CoreTensorImpl& A_; 
    const CoreTensorImpl& B_; 
    const std::vector<std::string>& Cinds_;
    const std::vector<std::string>& Ainds_;
    const std::vector<std::string>& Binds_;
    double alpha_;
    double beta_;

};

}

#endif
