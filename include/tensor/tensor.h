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
    using boost::shared_ptr;
    template<class T> using unique_ptr = boost::scoped_ptr<T>;

}

#endif

namespace tensor {

class TensorImpl;
class LabeledTensor;
class LabeledTensorProduct;

enum TensorType { Current , Core, Disk, Distributed, Agnostic };
enum EigenvalueOrder { Ascending, Descending };

typedef std::vector<size_t> Dimension;
typedef std::vector<std::pair<size_t, size_t> > IndexRange;
typedef std::vector<tuple<std::string, int, int, int> > ContractionTopology;

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
void finialize();

class Tensor {

public:

    // => Constructors <= //

    static Tensor build(TensorType type, const std::string& name, const Dimension& dims);

    static Tensor build(TensorType type, const Tensor& other);

    // => Reflectors <= //

    TensorType type() const;
    std::string name() const;
    const Dimension& dims() const;
    size_t rank() const;
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

    Tensor& scale_and_add(double a, const Tensor& x);
    Tensor& pointwise_multiplication(const Tensor& x);
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

private:

    shared_ptr<TensorImpl> tensor_;

protected:

    Tensor(shared_ptr<TensorImpl> tensor);

};

class LabeledTensor {

public:
    LabeledTensor(Tensor& T, const std::vector<std::string>& indices, double factor = 1.0) :
        T_(T), indices_(indices), factor_(factor)
    {}

    double factor() const { return factor_; }
    const std::vector<std::string>& indices() const { return indices_; }
    Tensor& T() const { return T_; }

    LabeledTensorProduct operator*(LabeledTensor& rhs);

    void operator=(LabeledTensor& rhs);
    void operator+=(LabeledTensor& rhs);
    void operator-=(LabeledTensor& rhs);

    void operator=(LabeledTensorProduct& rhs);
    void operator+=(LabeledTensorProduct& rhs);
    void operator-=(LabeledTensorProduct& rhs);

    void operator*=(double scale);

private:
    Tensor& T_;
    std::vector<std::string> indices_;
    double factor_;

};

inline LabeledTensor operator*(double factor, LabeledTensor& ti) {
    return LabeledTensor(ti.T(), ti.indices(), factor*ti.factor());
};

class LabeledTensorProduct {

public:
    LabeledTensorProduct(LabeledTensor& A, LabeledTensor& B) :
        A_(A), B_(B)
    {}

    LabeledTensor& A() const { return A_; }
    LabeledTensor& B() const { return B_; }

private:
    LabeledTensor& A_;
    LabeledTensor& B_;
};

}

#endif

