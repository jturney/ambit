#if !defined(TENSOR_TENSORIMPL_H)
#define TENSOR_TENSORIMPL_H

#include <string>
#include <vector>
#include <stdexcept>

#include <tensor/tensor.h>
#include "macros.h"

#include <boost/shared_ptr.hpp>

namespace tensor {

typedef TensorImpl* TensorImplPtr;
typedef TensorImpl const * ConstTensorImplPtr;

namespace detail {

class NotImplementedException : public std::logic_error
{
public:
    NotImplementedException(const std::string& str) : std::logic_error(str)
    {
    }
};

}

#define ThrowNotImplementedException throw detail::NotImplementedException(std::string("Function not yet implemented: ") + CURRENT_FUNCTION)

class OutOfMemoryException : public std::runtime_error
{
public:
    OutOfMemoryException() : std::runtime_error("Out of memory") {}
};

class TensorImpl
{
public:

    // => Constructors <= //

    TensorImpl(TensorType type, const std::string& name, const Dimension& dims)
        : type_(type), name_(name), dims_(dims)
    {}
    virtual ~TensorImpl() {}

    virtual void copy(ConstTensorImplPtr other);
    virtual TensorImplPtr clone(TensorType type = kCurrent);

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

    virtual void scale_and_add(double a, ConstTensorImplPtr x) = 0;
    virtual void pointwise_multiplication(ConstTensorImplPtr x) = 0;
    virtual void pointwise_division(ConstTensorImplPtr x) = 0;
    virtual double dot(ConstTensorImplPtr x) const = 0;

    // => Contraction Type Operations <= //

    virtual void contract(
        ConstTensorImplPtr A,
        ConstTensorImplPtr B,
        const ContractionTopology& topology,
        double alpha = 1.0,
        double beta = 0.0
        ) = 0;

    // => Rank-2 Operations <= //

    virtual std::map<std::string, TensorImplPtr> syev(EigenvalueOrder order) const = 0;
    virtual std::map<std::string, TensorImplPtr> geev(EigenvalueOrder order) const = 0;
    virtual std::map<std::string,TensorImplPtr> svd() const = 0;

    virtual TensorImplPtr cholesky() const = 0;
    virtual std::map<std::string, TensorImplPtr> lu() const = 0;
    virtual std::map<std::string, TensorImplPtr> qr() const = 0;

    virtual TensorImplPtr cholesky_inverse() const = 0;
    virtual TensorImplPtr inverse() const = 0;
    virtual TensorImplPtr power(double power, double condition = 1.0E-12) const = 0;

    virtual void givens(int dim, int i, int j, double s, double c) = 0;

protected:

    static bool typeCheck(TensorType type, ConstTensorImplPtr A, bool throwIfDiff = true);
    static bool rankCheck(size_t rank, ConstTensorImplPtr A, bool throwIfDiff = true);
    static bool squareCheck(ConstTensorImplPtr A, bool throwIfDiff = true);
    static bool dimensionCheck(ConstTensorImplPtr A, ConstTensorImplPtr B, bool throwIfDiff = true);

private:
    TensorType type_;
    std::string name_;
    Dimension dims_;
};

}

#endif

