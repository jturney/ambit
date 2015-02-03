#if !defined(TENSOR_INCLUDE_TENSOR_H)
#define TENSOR_INCLUDE_TENSOR_H

#include <cstdio>
#include <utility>
#include <vector>
#include <map>
#include <string>

#if defined(CXX11)
#include <memory>
#include <tuple>

namespace tensor {

    /*
     * If we have C++11 then we don't need Boost for shared_ptr, tuple, and unique_ptr.
     */
    using std::tuple;
    using std::shared_ptr;
    using std::unique_ptr;

}

#else
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/tuple/tuple.hpp>

namespace tensor {

    using boost::tuple;
    using boost::make_tuple;
    using boost::shared_ptr;
    template<class T> using unique_ptr = boost::scoped_ptr<T>;

}

#endif

namespace tensor {

static constexpr double numerical_zero__ = 1.0e-15;

class TensorImpl;
class LabeledTensor;
class LabeledTensorProduct;
class LabeledTensorAddition;
class LabeledTensorSubtraction;
class LabeledTensorDistributive;
class LabeledTensorSumOfProducts;

enum TensorType {
    kCurrent, kCore, kDisk, kDistributed, kAgnostic
};
enum EigenvalueOrder {
    kAscending, kDescending
};

typedef std::vector<size_t> Dimension;
typedef std::vector<std::pair<size_t, size_t> > IndexRange;
typedef std::vector<std::string> Indices;

/** Initializes the tensor library.
 *
 * Calls any necessary initialization of utilized frameworks.
 * @param argc number of command line arguments
 * @param argv the command line arguments
 * @return error code
 */
int initialize(int argc, char** argv);

/** Shutdowns the tensor library.
 *
 * Calls any necessary routines of utilized frameworks.
 */
void finalize();

class Tensor {

public:

    // => Constructors <= //

    static Tensor build(TensorType type, const std::string& name, const Dimension& dims);

    static Tensor build(TensorType type, const Tensor& other);

    void copy(const Tensor& other, const double& scale = 1.0);

    Tensor();

    // => Reflectors <= //

    TensorType type() const;
    std::string name() const;
    const Dimension& dims() const;
    size_t dim(size_t index) const;
    size_t rank() const;
    /// \return Total number of elements in the tensor.
    size_t numel() const;

    /**
     * Print some tensor information to fh
     * \param level If level = false, just print name and dimensions.  If level = true, print the entire tensor.
     **/
    void print(FILE* fh, bool level = false, const std::string& format = std::string("%11.6f"), int maxcols = 5) const;

    // => Labelers <= //

    LabeledTensor operator()(const std::string& indices);
    LabeledTensor operator[](const std::string& indices);

    // => Setters/Getters <= //

    void set_data(double* data, const IndexRange& ranges = IndexRange());
    void get_data(double* data, const IndexRange& ranges = IndexRange()) const;

    static double* get_block(const Tensor& tensor);
    static double* get_block(const IndexRange& ranges);
    static void free_block(double* data);

    // => Slicers <= //

    static Tensor slice(const Tensor& tensor, const IndexRange& ranges);
    static Tensor cat(const std::vector<Tensor>, int dim);

    // => Simple Single Tensor Operations <= //

    Tensor& zero();
    Tensor& scale(double a);
    double norm(double power = 2.0) const;

    // => Simple Double Tensor Operations <= //

    /**
    * Performs: C["ij"] += 2.0 * A["ij"];
    */
    Tensor& scale_and_add(const double& a, const Tensor& x);
    /**
    * Performs: C["ij"] *= A["ij"];
     */
    Tensor& pointwise_multiplication(const Tensor& x);
    /**
    * Performs: C["ij"] /= A["ij"];
    */
    Tensor& pointwise_division(const Tensor& x);
    double dot(const Tensor& x);

    // => Order-2 Operations <= //

    std::map<std::string, Tensor> syev(EigenvalueOrder order);
    std::map<std::string, Tensor> geev(EigenvalueOrder order);
    std::map<std::string, Tensor> svd();

    Tensor cholesky();
    std::map<std::string, Tensor> lu();
    std::map<std::string, Tensor> qr();

    Tensor cholesky_inverse();
    Tensor inverse();
    Tensor power(double power, double condition = 1.0E-12);

    Tensor& givens(int dim, int i, int j, double s, double c);

    // => Contraction Type Operations <= //

    void contract(
        const Tensor& A,
        const Tensor& B,
        const std::vector<std::string>& Cinds,
        const std::vector<std::string>& Ainds,
        const std::vector<std::string>& Binds,
        double alpha = 1.0,
        double beta = 1.0);

    void permute(
        const Tensor& A,
        const std::vector<std::string>& Cinds,
        const std::vector<std::string>& Ainds,
        double alpha = 1.0,
        double beta = 0.0);

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

private:

    shared_ptr<TensorImpl> tensor_;

protected:

    Tensor(shared_ptr<TensorImpl> tensor);

    std::map<std::string, Tensor> map_to_tensor(const std::map<std::string, TensorImpl*>& x);
};

class LabeledTensor {

public:
    LabeledTensor(Tensor T, const std::vector<std::string>& indices, double factor = 1.0);

    double factor() const { return factor_; }
    const Indices& indices() const { return indices_; }
    Tensor T() const { return T_; }

    LabeledTensorProduct operator*(const LabeledTensor& rhs);
    LabeledTensorAddition operator+(const LabeledTensor& rhs);
    LabeledTensorAddition operator-(const LabeledTensor& rhs);

    LabeledTensorDistributive operator*(const LabeledTensorAddition& rhs);

    /** Copies data from rhs to this sorting the data if needed. */
    void operator=(const LabeledTensor& rhs);
    void operator+=(const LabeledTensor& rhs);
    void operator-=(const LabeledTensor& rhs);
    void operator=(const LabeledTensorDistributive& rhs);
    void operator+=(const LabeledTensorDistributive& rhs);
    void operator-=(const LabeledTensorDistributive& rhs);

    void operator=(const LabeledTensorProduct& rhs);
    void operator+=(const LabeledTensorProduct& rhs);
    void operator-=(const LabeledTensorProduct& rhs);

    void operator=(const LabeledTensorAddition& rhs);
    void operator+=(const LabeledTensorAddition& rhs);
    void operator-=(const LabeledTensorAddition& rhs);

    void operator*=(const double& scale);
    void operator/=(const double& scale);

//    bool operator==(const LabeledTensor& other) const;
//    bool operator!=(const LabeledTensor& other) const;

    size_t numdim() const { return indices_.size(); }
    size_t dim_by_index(const std::string& idx) const;

    // negation
    LabeledTensor operator-() const {
        return LabeledTensor(T_, indices_, -factor_);
    }

private:

    void set(const LabeledTensor& to);

    Tensor T_;
    std::vector<std::string> indices_;
    double factor_;

};

inline LabeledTensor operator*(double factor, const LabeledTensor& ti) {
    return LabeledTensor(ti.T(), ti.indices(), factor*ti.factor());
};

class LabeledTensorProduct {

public:
    LabeledTensorProduct(const LabeledTensor& A, const LabeledTensor& B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    size_t size() const { return tensors_.size(); }

    const LabeledTensor& operator[](size_t i) const { return tensors_[i]; }

    LabeledTensorProduct& operator*(const LabeledTensor& other) {
        tensors_.push_back(other);
        return *this;
    }

    // conversion operator
    operator double() const;

    std::pair<double, double> compute_contraction_cost(const std::vector<size_t>& perm) const;

private:

    std::vector<LabeledTensor> tensors_;
};

class LabeledTensorAddition
{
public:
    LabeledTensorAddition(const LabeledTensor& A, const LabeledTensor& B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    size_t size() const { return tensors_.size(); }

    const LabeledTensor& operator[](size_t i) const { return tensors_[i]; }

    std::vector<LabeledTensor>::iterator begin() { return tensors_.begin(); }
    std::vector<LabeledTensor>::const_iterator begin() const { return tensors_.begin(); }

    std::vector<LabeledTensor>::iterator end() { return tensors_.end(); }
    std::vector<LabeledTensor>::const_iterator end() const { return tensors_.end(); }

    LabeledTensorAddition& operator+(const LabeledTensor& other) {
        tensors_.push_back(other);
        return *this;
    }

    LabeledTensorAddition& operator-(const LabeledTensor& other) {
        tensors_.push_back(-other);
        return *this;
    }

    LabeledTensorDistributive operator*(const LabeledTensor& other);

    LabeledTensorAddition& operator*(const double& scalar);

    // negation
    LabeledTensorAddition& operator-();

private:

    // This handles cases like T("ijab")
    std::vector<LabeledTensor> tensors_;

};

inline LabeledTensorAddition operator*(double factor, const LabeledTensorAddition& ti) {
    LabeledTensorAddition ti2 = ti;
    return ti2 * factor;
}

// Is responsible for expressions like D * (J - K) --> D*J - D*K
class LabeledTensorDistributive
{
public:
    LabeledTensorDistributive(const LabeledTensor& A, const LabeledTensorAddition& B)
            : A_(A), B_(B)
    {}

    const LabeledTensor& A() const { return A_; }
    const LabeledTensorAddition& B() const { return B_; }

private:

    const LabeledTensor& A_;
    const LabeledTensorAddition& B_;

};

}

#endif

