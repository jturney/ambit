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

#if !defined(TENSOR_CORE_H)
#define TENSOR_CORE_H

#include "tensor/tensorimpl.h"

namespace ambit
{

class CoreTensorImpl : public TensorImpl
{
  public:
    CoreTensorImpl(const string &name, const Dimension &dims);

    // Changes the internal dims_ object but does not change memory
    // allocation. This is an expert function. Used to change
    // the strides in the slice codes.
    void reshape(const Dimension &dims);

    // Changes the internal dims_ object and the memory allocation.
    void resize(const Dimension &dims, bool trim = true);

    vector<double> &data() { return data_; }
    const vector<double> &data() const { return data_; }

    double &at(const std::vector<size_t> &indices)
    {
        size_t pos = 0;
        for (int i = 0, maxi = rank(); i < maxi; i++)
        {
            pos += indices[i] * addressing()[i];
        }
        return data_[pos];
    }

    const double &at(const std::vector<size_t> &indices) const
    {
        size_t pos = 0;
        for (int i = 0, maxi = rank(); i < maxi; i++)
        {
            pos += indices[i] * addressing()[i];
        }
        return data_[pos];
    }

    // => Simple Single Tensor Operations <= //

    double norm(int type = 2) const;

    tuple<double, vector<size_t>> max() const;

    tuple<double, vector<size_t>> min() const;

    void scale(double beta = 0.0);

    void set(double alpha);

    void permute(ConstTensorImplPtr A, const Indices &Cinds,
                 const Indices &Ainds, double alpha = 1.0, double beta = 0.0);

    /**
     * Perform the following contraction using the TBLIS library
     *  C(Cinds) = alpha * A(Ainds) * B(Binds) + beta * C(Cinds)
     *
     * Parameters:
     *  @param A The left-side factor tensor, e.g., A2
     *  @param B The right-side factor tensor, e.g., B2
     *  @param Cinds The indices of tensor C, e.g., "ij"
     *  @param Ainds The indices of tensor A, e.g., "ik"
     *  @param Binds The indices of tensor B, e.g., "jk"
     *  @param alpha The scale applied to the product A*B, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     **/
    void contract(ConstTensorImplPtr A, ConstTensorImplPtr B,
                  const Indices &Cinds, const Indices &Ainds,
                  const Indices &Binds, double alpha = 1.0, double beta = 0.0);

    void contract(ConstTensorImplPtr A, ConstTensorImplPtr B,
                  const Indices &Cinds, const Indices &Ainds,
                  const Indices &Binds, std::shared_ptr<TensorImpl> &A2,
                  std::shared_ptr<TensorImpl> &B2,
                  std::shared_ptr<TensorImpl> &C2, double alpha = 1.0,
                  double beta = 0.0);

#if HAVE_TBLIS
    /**
     * Perform the following contraction using the TBLIS library
     *  C(Cinds) = alpha * A(Ainds) * B(Binds) + beta * C(Cinds)
     *
     * Parameters:
     *  @param A The left-side factor tensor, e.g., A2
     *  @param B The right-side factor tensor, e.g., B2
     *  @param Cinds The indices of tensor C, e.g., "ij"
     *  @param Ainds The indices of tensor A, e.g., "ik"
     *  @param Binds The indices of tensor B, e.g., "jk"
     *  @param alpha The scale applied to the product A*B, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     **/
    void contract_tblis(ConstTensorImplPtr A, ConstTensorImplPtr B,
                        const Indices &Cinds, const Indices &Ainds,
                        const Indices &Binds, double alpha, double beta);
#endif

    void gemm(ConstTensorImplPtr A, ConstTensorImplPtr B, bool transA,
              bool transB, size_t nrow, size_t ncol, size_t nzip, size_t ldaA,
              size_t ldaB, size_t ldaC, size_t offA = 0L, size_t offB = 0L,
              size_t offC = 0L, double alpha = 1.0, double beta = 0.0);

    // => Order-2 Operations <= //

    map<string, TensorImplPtr> syev(EigenvalueOrder order) const;
    map<string, TensorImplPtr> geev(EigenvalueOrder order) const;
    map<string, TensorImplPtr> gesvd() const;

    TensorImplPtr inverse() const;
    TensorImplPtr power(double power, double condition = 1.0E-12) const;

    void iterate(const function<void(const vector<size_t> &, double &)> &func);
    void citerate(const function<void(const vector<size_t> &, const double &)>
                      &func) const;

  private:
    vector<double> data_;
};

typedef CoreTensorImpl *CoreTensorImplPtr;
typedef const CoreTensorImpl *ConstCoreTensorImplPtr;
} // namespace ambit

#endif
