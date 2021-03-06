/*
 * @BEGIN LICENSE
 *
 * ambit: C++ library for the implementation of tensor product calculations
 *        through a clean, concise user interface.
 *
 * Copyright (c) 2014-2017 Ambit developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of ambit.
 *
 * Ambit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Ambit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ambit; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "tensorimpl.h"
#include "core/core.h"
#include "disk/disk.h"
#include "slice.h"
#include <numeric>

#if defined(HAVE_CYCLOPS)
#include "cyclops/cyclops.h"
#endif

namespace ambit
{

TensorImpl::TensorImpl(TensorType type, const string &name,
                       const Dimension &dims)
    : type_(type), name_(name), dims_(dims)
{
    numel_ = std::accumulate(dims_.begin(), dims_.end(), static_cast<size_t>(1),
                             std::multiplies<size_t>());

    addressing_ = Dimension(dims_.size(), 1);
    for (int n = static_cast<int>(dims_.size()) - 2; n >= 0; --n)
    {
        addressing_[n] = addressing_[n + 1] * dims_[n + 1];
    }
}

void TensorImpl::slice(ConstTensorImplPtr A, const IndexRange &Cinds,
                       const IndexRange &Ainds, double alpha, double beta)
{
    ambit::slice(this, A, Cinds, Ainds, alpha, beta);
}

void TensorImpl::zero() { scale(0.0); }

void TensorImpl::copy(ConstTensorImplPtr other)
{
    TensorImpl::dimensionCheck(this, other);

    IndexRange ranges;
    for (size_t ind = 0; ind < rank(); ind++)
    {
        ranges.push_back({0L, dims()[ind]});
    }
    slice(other, ranges, ranges, 1.0, 0.0);
}

TensorImplPtr TensorImpl::clone(TensorType t) const
{
    if (t == CurrentTensor)
    {
        t = type();
    }
    TensorImpl *tensor;
    if (t == CoreTensor)
    {
        tensor = new CoreTensorImpl(name(), dims());
    }
    else if (t == DiskTensor)
    {
        tensor = new DiskTensorImpl(name(), dims());
    }
#if defined(HAVE_ELEMENTAL)
    else if (t == DistributedTensor)
    {
        tensor = new cyclops::CyclopsTensorImpl(name(), dims());
    }
#endif
    else
    {
        throw std::runtime_error("TensorImpl::clone: Invalid TensorType");
    }
    tensor->copy(this);
    return tensor;
}

void TensorImpl::print(FILE *fh, bool level, const string & /*format*/,
                       int maxcols) const
{
    fprintf(fh, "  ## %s ##\n\n", name_.c_str());
    fprintf(fh, "  Rank = %zu\n", rank());
    fprintf(fh, "  Numel = %zu\n", numel());
    for (size_t dim = 0; dim < rank(); dim++)
    {
        fprintf(fh, "  Dimension %zu: %zu\n", dim + 1, dims_[dim]);
    }

    if (level)
    {
        double *temp;
        shared_ptr<TensorImpl> T;
        if (type() == CoreTensor)
        {
            temp = const_cast<double *>(data().data());
        }
        else
        {
            T = shared_ptr<TensorImpl>(clone(CoreTensor));
            temp = const_cast<double *>(T->data().data());
        }

        size_t order = rank();
        size_t nelem = numel();

        size_t page_size = 1L;
        size_t rows = 1;
        size_t cols = 1;
        if (order >= 1)
        {
            page_size *= dims_[order - 1];
            rows = dims_[order - 1];
        }
        if (order >= 2)
        {
            page_size *= dims_[order - 2];
            rows = dims_[order - 2];
            cols = dims_[order - 1];
        }

        fprintf(fh, "    Data:\n\n");

        if (nelem > 0)
        {
            size_t pages = nelem / page_size;
            for (size_t page = 0L; page < pages; page++)
            {

                if (order > 2)
                {
                    fprintf(fh, "    Page (");
                    size_t num = page;
                    size_t den = pages;
                    size_t val;
                    for (int k = 0; k < order - 2; k++)
                    {
                        den /= dims_[k];
                        val = num / den;
                        num -= val * den;
                        fprintf(fh, "%zu,", val);
                    }
                    fprintf(fh, "*,*):\n\n");
                }

                double *vp = temp + page * page_size;
                if (order == 0)
                {
                    fprintf(fh, "    %12.7f\n", *(vp));
                    fprintf(fh, "\n");
                }
                else if (order == 1)
                {
                    for (size_t i = 0; i < page_size; ++i)
                    {
                        fprintf(fh, "    %5zu %12.7f\n", i, *(vp + i));
                    }
                    fprintf(fh, "\n");
                }
                else
                {
                    for (size_t j = 0; j < cols;
                         j += static_cast<size_t>(maxcols))
                    {
                        size_t ncols = (j + static_cast<size_t>(maxcols) >= cols
                                            ? cols - j
                                            : static_cast<size_t>(maxcols));

                        // Column Header
                        fprintf(fh, "    %5s", "");
                        for (size_t jj = j; jj < j + ncols; jj++)
                        {
                            fprintf(fh, " %12zu", jj);
                        }
                        fprintf(fh, "\n");

                        // Data
                        for (size_t i = 0; i < rows; i++)
                        {
                            fprintf(fh, "    %5zu", i);
                            for (size_t jj = j; jj < j + ncols; jj++)
                            {
                                fprintf(fh, " %12.7f", *(vp + i * cols + jj));
                            }
                            fprintf(fh, "\n");
                        }

                        // Block separator
                        fprintf(fh, "\n");
                    }
                }
            }
        }
    }
}

bool TensorImpl::typeCheck(TensorType type, ConstTensorImplPtr A,
                           bool throwIfDiff)
{
    if (A->type() != type)
    {
        if (throwIfDiff)
        {
            throw std::runtime_error("TensorImpl::typeCheck: type mismatch");
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool TensorImpl::rankCheck(size_t rank, ConstTensorImplPtr A, bool throwIfDiff)
{
    if (A->rank() != rank)
    {
        if (throwIfDiff)
        {
            throw std::runtime_error("TensorImpl::rankCheck: Rank mismatch");
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool TensorImpl::squareCheck(ConstTensorImplPtr A, bool throwIfDiff)
{
    if (TensorImpl::rankCheck(2, A, throwIfDiff))
    {
        return true;
    }
    else
    {
        bool diff = (A->dims()[0] != A->dims()[1]);
        if (diff && throwIfDiff)
        {
            throw std::runtime_error(
                "TensorImpl::squareCheck: Dimension mismatch");
        }
        return diff;
    }
}

bool TensorImpl::dimensionCheck(ConstTensorImplPtr A, ConstTensorImplPtr B,
                                bool throwIfDiff)
{
    if (TensorImpl::rankCheck(A->rank(), B, throwIfDiff))
    {
        return true;
    }
    else
    {
        bool diff = false;
        int diffind = -1;
        for (size_t ind = 0; ind < A->rank(); ind++)
        {
            if (A->dims()[ind] != B->dims()[ind])
            {
                diff = true;
                diffind = ind;
                break;
            }
        }
        if (diff && throwIfDiff)
        {
            throw std::runtime_error(
                "TensorImpl::dimensionCheck: Dimension mismatch");
        } // Minor TODO
        return diff;
    }
}
} // namespace ambit
