#if !defined(TENSOR_TENSORIMPL_H)
#define TENSOR_TENSORIMPL_H

#include <string>
#include <vector>

#include <tensor/tensor.h>

namespace tensor {

class TensorImpl
{
public:

    // => Constructors <= //

    TensorImpl(TensorType type, const std::string& name, const Dimension& dims)
        : type_(type), name_(name), dims_(dims)
    {}
    virtual ~TensorImpl() {}

    virtual void copy(const TensorImpl& other);
    virtual TensorImpl* clone(TensorType type = Current);

    // => Reflectors <= //

    virtual TensorType type() const { return type_; }
    std::string name() const { return name_; }
    virtual const Dimension& dims() const { return dims_; }
    virtual size_t rank() const { return dims_.size(); }
    size_t numel() const;

    /**
     * Print some tensor information to fh
     * \param level If level = false, just print name and dimensions.  If level = true, print the entire tensor.
     **/
    void print(FILE* fh, bool level = false, const std::string& format = "%12.7f", int maxcols = 5) const;

    // => Setters/Getters <= //

    static double* get_block(size_t numel);
    static double* free_block(double* data);

    virtual void set_data(double* data, const IndexRange& ranges = IndexRange()) = 0;
    virtual void get_data(double* data, const IndexRange& ranges = IndexRange()) const = 0;

    // => Simple Single Tensor Operations <= //

    virtual void zero() = 0;
    virtual void scale(double a) = 0;
    virtual double norm(double power = 2.0) const = 0;
    virtual double rms(double power = 2.0) const = 0;

    // => Simple Double Tensor Operations <= //

    virtual void scale_and_add(double a, const TensorImpl& x) = 0;
    virtual void pointwise_multiplication(const TensorImpl& x) = 0;
    virtual void pointwise_division(const TensorImpl& x) = 0;
    virtual double dot(const TensorImpl& x) const = 0;

    // => Contraction Type Operations <= //

    virtual void contract(
        const TensorImpl& A,
        const TensorImpl& B,
        const ContractionTopology& topology,
        double alpha = 1.0,
        double beta = 0.0
        ) = 0;

    // => Rank-2 Operations <= //

    virtual std::map<std::string, TensorImpl*> syev(EigenvalueOrder order) const = 0;
    virtual std::map<std::string, TensorImpl*> geev(EigenvalueOrder order) const = 0;
    virtual std::map<std::string, TensorImpl*> svd() const = 0;

    virtual TensorImpl* cholesky() const = 0;
    virtual std::map<std::string, TensorImpl*> lu() const = 0;
    virtual std::map<std::string, TensorImpl*> qr() const = 0;

    virtual TensorImpl* cholesky_inverse() const = 0;
    virtual TensorImpl* inverse() const = 0;
    virtual TensorImpl* power(double power, double condition = 1.0E-12) const = 0;

    virtual void givens(int dim, int i, int j, double s, double c) = 0;

protected:

    static bool typeCheck(TensorType type, const TensorImpl& A, bool throwIfDiff = true);
    static bool rankCheck(size_t rank, const TensorImpl& A, bool throwIfDiff = true);
    static bool squareCheck(const TensorImpl& A, bool throwIfDiff = true);
    static bool dimensionCheck(const TensorImpl& A, const TensorImpl& B, bool throwIfDiff = true);

private:
    TensorType type_;
    std::string name_;
    Dimension dims_;
};

}

#endif

