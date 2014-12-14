#include "tensorimpl.h"
#include "core.h"

namespace tensor {

void TensorImpl::copy(ConstTensorImplPtr other)
{
    TensorImpl::dimensionCheck(this, other);
    double* temp = TensorImpl::get_block(numel());
    other->get_data(temp);
    set_data(temp);
    temp = TensorImpl::free_block(temp);
}
TensorImplPtr TensorImpl::clone(TensorType t)
{
    if (t == Current) {
        t = type();
    }
    TensorImpl* tensor;
    if (t == Core) {
        tensor = new CoreTensorImpl(name(), dims());
    } else {
        throw std::runtime_error("TensorImpl::clone: Invalid TensorType");
    }
    tensor->copy(this);
    return tensor;
}
size_t TensorImpl::numel() const
{
    size_t numel = 1L;
    for (size_t ind = 0; ind < dims_.size(); ind++) {
        numel *= dims_[ind];
    }
    return numel;
}
void TensorImpl::print(FILE* fh, bool level, const std::string& /*format*/, int maxcols) const
{
    fprintf(fh, "  ## %s ##\n\n", name_.c_str());
    fprintf(fh, "  Rank = %zu\n", rank());
    fprintf(fh, "  Numel = %zu\n", numel());
    for (size_t dim = 0; dim < rank(); dim++) {
        fprintf(fh, "  Dimension %zu: %zu\n", dim+1, dims_[dim]);
    }

    if (level > 0) {
        double* temp = get_block(numel());
        get_data(temp);

        int order = rank();
        size_t nelem = numel();

        size_t page_size = 1L;
        size_t rows = 1;
        size_t cols = 1;
        if (order >= 1) {
            page_size *= dims_[order - 1];
            rows = dims_[order - 1];
        }
        if (order >= 2) {
            page_size *= dims_[order - 2];
            rows = dims_[order - 2];
            cols = dims_[order - 1];
        }

        fprintf(fh, "    Data:\n\n");

        size_t pages = nelem / page_size;
        for (size_t page = 0L; page < pages; page++) {

            if (order > 2) {
                fprintf(fh, "    Page (");
                size_t num = page;
                size_t den = pages;
                size_t val;
                for (int k = 0; k < order - 2; k++) {
                    den /= dims_[k];
                    val = num / den;
                    num -= val * den;
                    fprintf(fh,"%zu,",val);
                }
                fprintf(fh, "*,*):\n\n");
            }

            double* vp = temp + page * page_size;
            if (order == 0) {
                fprintf(fh, "    %12.7f\n", *(vp));
                fprintf(fh,"\n");
            } else if(order == 1) {
                for (size_t i=0; i<page_size; ++i) {
                    fprintf(fh, "    %5zu %12.7f\n", i, *(vp + i));
                }
                fprintf(fh,"\n");
            } else {
                for (size_t j = 0; j < cols; j+= maxcols) {
                    size_t ncols = (j + maxcols >= cols ? cols - j : maxcols);

                    // Column Header
                    fprintf(fh,"    %5s", "");
                    for (size_t jj = j; jj < j+ncols; jj++) {
                        fprintf(fh," %12zu", jj);
                    }
                    fprintf(fh,"\n");

                    // Data
                    for (size_t i = 0; i < rows; i++) {
                        fprintf(fh,"    %5zu", i);
                        for (size_t jj = j; jj < j+ncols; jj++) {
                            fprintf(fh," %12.7f", *(vp + i * cols + jj));
                        }
                        fprintf(fh,"\n");
                    }

                    // Block separator
                    fprintf(fh,"\n");
                }
            }
        }
        temp = free_block(temp);
    }
}
double* TensorImpl::get_block(size_t numel)
{
    return new double[numel];
}
double* TensorImpl::free_block(double* numel)
{
    delete[] numel;
    return NULL;
}
bool TensorImpl::typeCheck(TensorType type, ConstTensorImplPtr A, bool throwIfDiff)
{
    if (A->type() != type) {
        if (throwIfDiff) throw std::runtime_error("TensorImpl::typeCheck: type mismatch");
        return true;
    } else {
        return false;
    }
}
bool TensorImpl::rankCheck(size_t rank, ConstTensorImplPtr A, bool throwIfDiff)
{
    if (A->rank() != rank) {
        if (throwIfDiff) throw std::runtime_error("TensorImpl::rankCheck: Rank mismatch");
        return true;
    } else {
        return false;
    }
}
bool TensorImpl::squareCheck(ConstTensorImplPtr A, bool throwIfDiff)
{
    if (TensorImpl::rankCheck(2, A, throwIfDiff)) {
        return true;
    } else {
        bool diff = (A->dims()[0] != A->dims()[1]);
        if (diff && throwIfDiff) throw std::runtime_error("TensorImpl::squareCheck: Dimension mismatch");
        return diff;
    }
}
bool TensorImpl::dimensionCheck(ConstTensorImplPtr A, ConstTensorImplPtr B, bool throwIfDiff)
{
    if (TensorImpl::rankCheck(A->rank(), B, throwIfDiff)) {
        return true;
    } else {
        bool diff = false;
        int diffind = -1;
        for (size_t ind = 0; ind < A->rank(); ind++) {
            if (A->dims()[ind] != B->dims()[ind]) {
                diff = true;
                diffind = ind;
                break;
            }
        }
        if (diff && throwIfDiff) throw std::runtime_error("TensorImpl::dimensionCheck: Dimension mismatch"); // Minor TODO
        return diff;
    }
}



}
