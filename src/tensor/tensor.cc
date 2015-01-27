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

}

// Constructs a ContractionTopology object for C = A * B.
ContractionTopology::ContractionTopology(const LabeledTensor& C, const LabeledTensor& A, const LabeledTensor& B)
{
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

        indices_.push_back(*index);
        A_pos_.push_back(Apos);
        B_pos_.push_back(Bpos);
        C_pos_.push_back(Cpos);

        ContractionType type;
        if (Cpos != -1 && Apos != -1 && Bpos != -1) {
            type = ABC;
            if (A.T().dims()[Apos] != B.T().dims()[Bpos] || A.T().dims()[Apos] != C.T().dims()[Cpos]) {
                throw std::runtime_error("Invalid ABC contraction index sizes!");
            }
        } else if (Apos != -1 && Bpos != -1 && Cpos == -1) {
            type = AB;
            if (A.T().dims()[Apos] != B.T().dims()[Bpos]) {
                throw std::runtime_error("Invalid AB contraction index sizes!");
            }
        } else if (Apos != -1 && Bpos == -1 && Cpos != -1) {
            type = AC;
            if (A.T().dims()[Apos] != C.T().dims()[Cpos]) {
                throw std::runtime_error("Invalid AC contraction index sizes!");
            }
        } else if (Apos == -1 && Bpos != -1 && Cpos != -1) {
            type = BC;
            if (B.T().dims()[Bpos] != C.T().dims()[Cpos]) {
                throw std::runtime_error("Invalid BC contraction index sizes!");
            }
        } else
            throw std::runtime_error("Invalid contraction topology!");

        printf("%s: %d %d %d %d\n", index->c_str(), Cpos, Apos, Bpos, type);
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

Tensor::Tensor(shared_ptr<TensorImpl> tensor)
    : tensor_(tensor)
{}

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
    tensor_->set_data(data, ranges);
}

void Tensor::get_data(double *data, IndexRange const &ranges) const
{
    tensor_->get_data(data, ranges);
}

double* Tensor::get_block(const Tensor& tensor)
{
    return TensorImpl::get_block(tensor.numel());
}

double* Tensor::get_block(const IndexRange &ranges)
{
    size_t nel = 1;
    for (IndexRange::const_iterator iter = ranges.begin();
            iter != ranges.end();
            ++iter) {
        nel *= iter->second - iter->first;
    }
    return TensorImpl::get_block(nel);
}

void Tensor::free_block(double *data)
{
    TensorImpl::free_block(data);
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
    tensor_->zero();
}

Tensor& Tensor::scale(double a)
{
    tensor_->scale(a);
    return *this;
}

double Tensor::norm(double power) const
{
    return tensor_->norm(power);
}

Tensor& Tensor::scale_and_add(double a, const Tensor &x)
{
    tensor_->scale_and_add(a, x.tensor_.get());
    return *this;
}

Tensor& Tensor::pointwise_multiplication(const Tensor &x)
{
    tensor_->pointwise_multiplication(x.tensor_.get());
    return *this;
}

Tensor& Tensor::pointwise_division(const Tensor &x)
{
    tensor_->pointwise_division(x.tensor_.get());
    return *this;
}

double Tensor::dot(const Tensor& x)
{
    return tensor_->dot(x.tensor_.get());
}

std::map<std::string, Tensor> Tensor::map_to_tensor(const std::map<std::string, TensorImplPtr>& x)
{
    std::map<std::string, Tensor> result;

    for (std::map<std::string, TensorImplPtr>::const_iterator iter = x.begin();
            iter != x.end();
            ++iter) {
        result.insert(make_pair(iter->first, Tensor(shared_ptr<TensorImpl>(iter->second))));
    }
    return result;
}

std::map<std::string, Tensor> Tensor::syev(EigenvalueOrder order)
{
    return map_to_tensor(tensor_->syev(order));
}

std::map<std::string, Tensor> Tensor::geev(EigenvalueOrder order)
{
    return map_to_tensor(tensor_->geev(order));
}

std::map<std::string, Tensor> Tensor::svd()
{
    return map_to_tensor(tensor_->svd());
}

Tensor Tensor::cholesky()
{
    return Tensor(shared_ptr<TensorImpl>(tensor_->cholesky()));
}

std::map<std::string, Tensor> Tensor::lu()
{
    return map_to_tensor(tensor_->lu());
}

std::map<std::string, Tensor> Tensor::qr()
{
    return map_to_tensor(tensor_->qr());
}

Tensor Tensor::cholesky_inverse()
{
    return Tensor(shared_ptr<TensorImpl>(tensor_->cholesky_inverse()));
}

Tensor Tensor::inverse()
{
    return Tensor(shared_ptr<TensorImpl>(tensor_->inverse()));
}

Tensor Tensor::power(double power, double condition)
{
    return Tensor(shared_ptr<TensorImpl>(tensor_->power(power, condition)));
}

Tensor& Tensor::givens(int dim, int i, int j, double s, double c)
{
    tensor_->givens(dim, i, j, s, c);
    return *this;
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
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    // 1. create a ContractionTopology
    ContractionTopology ct(*this, A, B);

    // 2. call contract on the tensor.
    T_.contract(A.T(),
                B.T(),
                ct,
                A.factor() * B.factor(),
                0.0);
}

void LabeledTensor::operator+=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    // 1. create a ContractionTopology
    ContractionTopology ct(*this, A, B);

    // 2. call contract on the tensor.
    T_.contract(A.T(),
                B.T(),
                ct,
                A.factor() * B.factor(),
                1.0);
}

void LabeledTensor::operator-=(const LabeledTensorProduct& rhs)
{
    // Perform a tensor contraction.
    assert(rhs.size() == 2);
    const LabeledTensor& A = rhs[0];
    const LabeledTensor& B = rhs[1];

    // 1. create a ContractionTopology
    ContractionTopology ct(*this, A, B);

    // 2. call contract on the tensor.
    T_.contract(A.T(),
                B.T(),
                ct,
                - A.factor() * B.factor(),
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
