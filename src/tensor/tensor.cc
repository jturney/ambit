#include <tensor/tensor.h>
#include "tensorimpl.h"
#include "core.h"

// include header files to specific tensor types supported.
#if defined(HAVE_CYCLOPS)
#   include "cyclops/cyclops.h"
#endif

#include <sstream>

namespace tensor {

namespace util {

namespace {

// trim from start
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return util::ltrim(util::rtrim(s));
}

}

std::vector<std::string> split_indices(const std::string &indices)
{
    std::istringstream f(indices);
    std::string s;
    std::vector<std::string> v;

    if (indices.find(",") != std::string::npos) {
        while (std::getline(f, s, ',')) {
            std::string trimmed = util::trim(s);
            v.push_back(trimmed);
        }
    }
    else {
        // simply split the string up
        for (int i=0; i<indices.size(); ++i)
            v.push_back(std::string(1, indices[i]));
    }

    return v;
}

}

int initialize(int argc, char** argv)
{
#if defined(HAVE_CYCLOPS)
    cyclops::initialize(argc, argv);
#endif

    return 0;
}

void finialize()
{
#if defined(HAVE_CYCLOPS)
    cyclops::finalize();
#endif
}

Tensor Tensor::build(TensorType type, const std::string& name, const Dimension& dims)
{
    Tensor newObject;

    if (type == kAgnostic) {
        #if defined(HAVE_CYCLOPS)
        type = kDistributed;
        #else
        type = kCore;
        #endif
    }
    switch(type) {
        case kCore:
            newObject.tensor_.reset(new CoreTensorImpl(name, dims));
            break;

        case kDisk:
            // TODO: Construct disk tensor object
            break;

        case kDistributed:

            #if defined(HAVE_CYCLOPS)
            newObject.tensor_.reset(new cyclops::CyclopsTensorImpl(name, dims));
            #else
            throw std::runtime_error("Tensor::build: Unable to construct distributed tensor object");
            #endif

            break;

        default:
            throw std::runtime_error("Tensor::build: Unknown parameter passed into 'type'.");
    }
}

Tensor Tensor::build(TensorType type, const Tensor& other)
{
    ThrowNotImplementedException;
}

Tensor::Tensor()
{}

TensorType Tensor::type() const
{
    return tensor_->type();
}

std::string Tensor::name() const
{
    return tensor_->name();
}

const std::vector<size_t>& Tensor::dims() const
{
    return tensor_->dims();
}

size_t Tensor::rank() const
{
    return tensor_->dims().size();
}

size_t Tensor::numel() const
{
    return tensor_->numel();
}

void Tensor::print(FILE *fh, bool level, std::string const &format, int maxcols) const
{
    ThrowNotImplementedException;
}

LabeledTensor Tensor::operator()(const std::string& indices)
{
    return LabeledTensor(*this, util::split_indices(indices));
}

LabeledTensor Tensor::operator[](const std::string& indices)
{
    return LabeledTensor(*this, util::split_indices(indices));
}

void Tensor::set_data(double *data, IndexRange const &ranges)
{
    ThrowNotImplementedException;
}

void Tensor::get_data(double *data, IndexRange const &ranges) const
{
    ThrowNotImplementedException;
}

double* Tensor::get_block(const Tensor& tensor)
{
    ThrowNotImplementedException;
}

double* Tensor::get_block(const IndexRange &ranges)
{
    ThrowNotImplementedException;
}

void Tensor::free_block(double *data)
{
    ThrowNotImplementedException;
}

Tensor Tensor::slice(const Tensor &tensor, const IndexRange &ranges)
{
    ThrowNotImplementedException;
}

Tensor Tensor::cat(std::vector<Tensor> const, int dim)
{
    ThrowNotImplementedException;
}

Tensor& Tensor::zero()
{
    ThrowNotImplementedException;
}

Tensor& Tensor::scale(double a)
{
    ThrowNotImplementedException;
}

double Tensor::norm(double power) const
{
    ThrowNotImplementedException;
}

Tensor& Tensor::scale_and_add(double a, const Tensor &x)
{
    ThrowNotImplementedException;
}

Tensor& Tensor::pointwise_multiplication(const Tensor &x)
{
    ThrowNotImplementedException;
}

Tensor& Tensor::pointwise_division(const Tensor &x)
{
    ThrowNotImplementedException;
}

double Tensor::dot(const Tensor& x)
{
    ThrowNotImplementedException;
}

std::map<std::string, Tensor> Tensor::syev(EigenvalueOrder order)
{
    ThrowNotImplementedException;

}
std::map<std::string, Tensor> Tensor::geev(EigenvalueOrder order)
{
    ThrowNotImplementedException;

}
std::map<std::string, Tensor> Tensor::svd()
{
    ThrowNotImplementedException;

}

Tensor Tensor::cholesky()
{
    ThrowNotImplementedException;

}
std::map<std::string, Tensor> Tensor::lu()
{
    ThrowNotImplementedException;

}
std::map<std::string, Tensor> Tensor::qr()
{
    ThrowNotImplementedException;

}

Tensor Tensor::cholesky_inverse()
{
    ThrowNotImplementedException;

}
Tensor Tensor::inverse()
{
    ThrowNotImplementedException;

}
Tensor Tensor::power(double power, double condition)
{
    ThrowNotImplementedException;

}

Tensor& Tensor::givens(int dim, int i, int j, double s, double c)
{
    ThrowNotImplementedException;
}

/********************************************************************
* LabeledTensor operators
********************************************************************/
void LabeledTensor::operator=(const LabeledTensor& rhs)
{

}

}
