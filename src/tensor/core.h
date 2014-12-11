#if !defined(TENSOR_CORE_H)
#define TENSOR_CORE_H

namespace tensor {

class CoreTensorImpl : public TensorImpl
{
public:
    CoreTensorImpl(const std::string& name, const Dimension& dims);
    ~CoreTensorImpl();

    void set_data(double* data, const IndexRange& ranges = IndexRange());
    void get_data(double* data, const IndexRange& ranges = IndexRange()) const;

    // => Simple Single Tensor Operations <= //

    void zero();
    void scale(double a);
    double norm(double power = 2.0) const;
    double rms(double power = 2.0) const;

    // => Simple Double TensorImpl Operations <= //

    void scale_and_add(double a, const TensorImpl& x);
    void pointwise_multiplication(const TensorImpl& x);
    void pointwise_division(const TensorImpl& x);
    double dot(const TensorImpl& x) const;

    // => Contraction Type Operations <= //

    void contract(
         const TensorImpl& A,
         const TensorImpl& B,
         const ContractionTopology& topology,
         double alpha = 1.0,
         double beta = 0.0
         );

    // => Order-2 Operations <= //

    std::map<std::string, TensorImpl*> syev(EigenvalueOrder order) const;
    std::map<std::string, TensorImpl*> geev(EigenvalueOrder order) const;
    std::map<std::string, TensorImpl*> svd() const;

    TensorImpl* cholesky() const;
    std::map<std::string, TensorImpl*> lu() const;
    std::map<std::string, TensorImpl*> qr() const;

    TensorImpl* cholesky_inverse() const;
    TensorImpl* inverse() const;
    TensorImpl* power(double power, double condition = 1.0E-12) const;

    void givens(int dim, int i, int j, double s, double c);
    
private:
    double* data_;    
};

}

#endif
