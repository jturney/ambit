#if !defined(TENSOR_TENSORIMPL_H)
#define TENSOR_TENSORIMPL_H

#include <string>
#include <vector>
#include <stdexcept>

#include <ambit/tensor.h>
#include "macros.h"

namespace ambit {

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

class OutOfMemoryException : public std::runtime_error
{
public:
    OutOfMemoryException() : std::runtime_error("Out of memory") {}
};

}

#define ThrowNotImplementedException throw detail::NotImplementedException(std::string("Function not yet implemented: ") + CURRENT_FUNCTION)

class TensorImpl
{
public:

    // => Constructors <= //

    TensorImpl(TensorType type, const std::string& name, const Dimension& dims);
    virtual ~TensorImpl() {}

    virtual TensorImplPtr clone(TensorType type = kCurrent) const;

    // => Reflectors <= //

    TensorType type() const { return type_; }
    std::string name() const { return name_; }
    const Dimension& dims() const { return dims_; }
    size_t dim(size_t ind) const { return dims_[ind]; }
    size_t rank() const { return dims_.size(); }
    size_t numel() const { return numel_; }

    void set_name(const std::string& name) { name_ = name; }

    void print(FILE* fh = stdout, bool level = true, const std::string& format = "%12.7f", int maxcols = 5) const;

    // => Setters/Getters <= //

    virtual std::vector<double>& data()
        { throw std::runtime_error("TensorImpl::data() not supported for tensor type " + std::to_string(type())); }
    virtual const std::vector<double>& data() const
        { throw std::runtime_error("TensorImpl::data() not supported for tensor type " + std::to_string(type())); }

    // => Simple Single Tensor Operations <= //

    virtual double norm(
        int type = 2) const
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }

    virtual std::tuple<double, std::vector<size_t>> max() const
        { throw std::runtime_error("Operation not support in this tensor implementation."); }

    virtual std::tuple<double, std::vector<size_t>> min() const
        { throw std::runtime_error("Operation not support in this tensor implementation."); }

    void zero();

    virtual void scale(
        double beta = 0.0)
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }

    virtual void set(double alpha)
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }

    void copy(
        ConstTensorImplPtr other);

    virtual void slice(
        ConstTensorImplPtr A,
        const IndexRange& Cinds,
        const IndexRange& Ainds,
        double alpha = 1.0,
        double beta = 0.0);

    virtual void permute(
        ConstTensorImplPtr A,
        const Indices& Cinds,
        const Indices& Ainds,
        double alpha = 1.0,
        double beta = 0.0)
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }

    virtual void contract(
        ConstTensorImplPtr A,
        ConstTensorImplPtr B,
        const Indices& Cinds,
        const Indices& Ainds,
        const Indices& Binds,
        double alpha = 1.0,
        double beta = 0.0)
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }

    virtual void gemm(
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
        double beta = 0.0)
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }

    // => Rank-2 Operations <= //

    virtual std::map<std::string, TensorImplPtr> syev(EigenvalueOrder order) const
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }
    virtual std::map<std::string, TensorImplPtr> geev(EigenvalueOrder order) const
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }

    //virtual std::map<std::string, TensorImplPtr> svd() const = 0;

    //virtual TensorImplPtr cholesky() const = 0;
    //virtual std::map<std::string, TensorImplPtr> lu() const = 0;
    //virtual std::map<std::string, TensorImplPtr> qr() const = 0;

    //virtual TensorImplPtr cholesky_inverse() const = 0;
    //virtual TensorImplPtr inverse() const = 0;
    virtual TensorImplPtr power(double power, double condition = 1.0E-12) const
        { throw std::runtime_error("Operation not supported in this tensor implementation."); }

    // => Iterators <= //

    virtual void iterate(const std::function<void (const std::vector<size_t>&, double&)>& func)
    { throw std::runtime_error("Operation not supported in this tensor implementation."); }
    virtual void citerate(const std::function<void (const std::vector<size_t>&, const double&)>& func) const
    { throw std::runtime_error("Operation not supported in this tensor implementation."); }


protected:

    static bool typeCheck(TensorType type, ConstTensorImplPtr A, bool throwIfDiff = true);
    static bool rankCheck(size_t rank, ConstTensorImplPtr A, bool throwIfDiff = true);
    static bool squareCheck(ConstTensorImplPtr A, bool throwIfDiff = true);
    static bool dimensionCheck(ConstTensorImplPtr A, ConstTensorImplPtr B, bool throwIfDiff = true);

private:
    TensorType type_;
    std::string name_;
    Dimension dims_;
    size_t numel_;
};

}

#endif

