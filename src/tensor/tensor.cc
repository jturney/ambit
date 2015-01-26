#include <tensor/tensor.h>
#include "tensorimpl.h"
#include "core.h"
#include "indices.h"

// include header files to specific tensor types supported.
#if defined(HAVE_CYCLOPS)
#   include "cyclops/cyclops.h"
#endif

#include <list>
#include <algorithm>

namespace tensor {

namespace {

void add_all_indices(const LabeledTensor& x, std::list<std::string>& list)
{
    std::vector<std::string>::const_iterator iter, end = x.indices().end();
    for (iter = x.indices().begin(); iter != end; ++iter) {
        list.push_back(*iter);
    }
}

int find_index_in_labeled_tensor(const LabeledTensor& x, const std::string& index)
{
    long xpos = -1;
    std::vector<std::string>::const_iterator at = std::find(x.indices().begin(), x.indices().end(), index);
    if (at != x.indices().end()) {
        xpos = at - x.indices().begin();
    }

    return (int)xpos;
}

// Constructs a ContractionTopology object for C = A * B.
ContractionTopology make_contraction_topology(const LabeledTensor& C, const LabeledTensor& A, const LabeledTensor& B)
{
    ContractionTopology newTopology;
    std::list<std::string> listOfIndices;

    // Add all indices from each LabeledTensor
    add_all_indices(C, listOfIndices);
    add_all_indices(A, listOfIndices);
    add_all_indices(B, listOfIndices);

    // Sort and grab unique
    listOfIndices.sort();
    listOfIndices.unique();

    printf("Contraction Topology:\n");
    // Walk our way through the list of unique indices and assign topology values
    for (std::list<std::string>::const_iterator index = listOfIndices.begin(),
            end = listOfIndices.end();
            index != end;
            ++index) {

        // In each LabeledTensor find the index
        int Cpos = find_index_in_labeled_tensor(C, *index),
            Apos = find_index_in_labeled_tensor(A, *index),
            Bpos = find_index_in_labeled_tensor(B, *index);

        newTopology.push_back(boost::make_tuple(*index, Cpos, Apos, Bpos));

        printf("%s: %d %d %d\n", index->c_str(), Cpos, Apos, Bpos);
    }

    return newTopology;
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
//            printf("Constructing core tensor.\n");
            newObject.tensor_.reset(new CoreTensorImpl(name, dims));
            break;

        case kDisk:
//            printf("Constructing disk tensor.\n");
            // TODO: Construct disk tensor object
            break;

        case kDistributed:
//            printf("Constructing distributed tensor.\n");

            #if defined(HAVE_CYCLOPS)
            newObject.tensor_.reset(new cyclops::CyclopsTensorImpl(name, dims));
            #else
            throw std::runtime_error("Tensor::build: Unable to construct distributed tensor object");
            #endif

            break;

        default:
            throw std::runtime_error("Tensor::build: Unknown parameter passed into 'type'.");
    }

    return newObject;
}

Tensor Tensor::build(TensorType type, const Tensor& other)
{
    ThrowNotImplementedException;
}

void Tensor::copy(const Tensor& other, const double& scale)
{
    tensor_->copy(other.tensor_.get(), scale);
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
    tensor_->print(fh, level, format, maxcols);
}

LabeledTensor Tensor::operator()(const std::string& indices)
{
    return LabeledTensor(*this, indices::split(indices));
}

LabeledTensor Tensor::operator[](const std::string& indices)
{
    return LabeledTensor(*this, indices::split(indices));
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

void Tensor::contract(const Tensor &A, const Tensor &B, const ContractionTopology &topology, double alpha, double beta)
{
    tensor_->contract(A.tensor_.get(),
                      B.tensor_.get(),
                      topology,
                      alpha,
                      beta);
}

/********************************************************************
* LabeledTensor operators
********************************************************************/
void LabeledTensor::operator=(const LabeledTensor& rhs)
{
    if (indices::equivalent(indices_, rhs.indices_) == true) {
        // equivalent indices:   "i,a" = "i,a"
        // perform a simple copy
        T_.copy(rhs.T(), rhs.factor());
    }
    else {
        // TODO: potential sorting of data
        printf("Potential sorting assignment.\n");
        ThrowNotImplementedException;
    }
}

LabeledTensorProduct LabeledTensor::operator*(const LabeledTensor &rhs)
{
    return LabeledTensorProduct(*this, rhs);
}

LabeledTensorAddition LabeledTensor::operator+(const LabeledTensor &rhs)
{
    return LabeledTensorAddition(*this, rhs);
}

LabeledTensorSubtraction LabeledTensor::operator-(const LabeledTensor &rhs)
{
    return LabeledTensorSubtraction(*this, rhs);
}

void LabeledTensor::operator=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.

    // 1. create a ContractionTopology
    ContractionTopology ct = make_contraction_topology(*this,
                                                       rhs.A(),
                                                       rhs.B());

    // 2. call contract on the tensor.
    T_.contract(rhs.A().T(),
                rhs.B().T(),
                ct,
                rhs.A().factor() * rhs.B().factor(),
                0.0);
}

void LabeledTensor::operator+=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.

    // 1. create a ContractionTopology
    ContractionTopology ct = make_contraction_topology(*this,
                                                       rhs.A(),
                                                       rhs.B());

    // 2. call contract on the tensor.
    T_.contract(rhs.A().T(),
                rhs.B().T(),
                ct,
                rhs.A().factor() * rhs.B().factor(),
                1.0);
}

void LabeledTensor::operator-=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.

    // 1. create a ContractionTopology
    ContractionTopology ct = make_contraction_topology(*this,
                                                       rhs.A(),
                                                       rhs.B());

    // 2. call contract on the tensor.
    T_.contract(rhs.A().T(),
                rhs.B().T(),
                ct,
                - rhs.A().factor() * rhs.B().factor(),
                1.0);
}

void LabeledTensor::operator=(const LabeledTensorAddition& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator+=(const LabeledTensorAddition& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator-=(const LabeledTensorAddition& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator=(const LabeledTensorSubtraction& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator+=(const LabeledTensorSubtraction& rhs)
{
    ThrowNotImplementedException;
}

void LabeledTensor::operator-=(const LabeledTensorSubtraction& rhs)
{
    ThrowNotImplementedException;
}

}
