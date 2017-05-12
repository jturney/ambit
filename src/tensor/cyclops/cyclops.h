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
 * You should have received a copy of the GNU Lesser General Public License along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#if !defined(TENSOR_CYCLOPS_H)
#define TENSOR_CYCLOPS_H

#include "tensor/tensorimpl.h"
#include <ctf.hpp>
#include <El.hpp>

namespace ambit
{

namespace cyclops
{

int initialize(int argc, char **argv);

int initialize(MPI_Comm comm, int argc = 0, char **argv = nullptr);

void finalize();

class CyclopsTensorImpl : public TensorImpl
{
  public:
    CyclopsTensorImpl(const std::string &name, const Dimension &dims);
    ~CyclopsTensorImpl();

    CTF_Tensor *cyclops() const { return cyclops_; }

    // => Simple Single Tensor Operations <= //

    double norm(int type = 2) const;

    std::tuple<double, std::vector<size_t>> max() const;

    std::tuple<double, std::vector<size_t>> min() const;

    void scale(double beta = 0.0);

    void set(double alpha);

    void permute(ConstTensorImplPtr A, const std::vector<std::string> &Cinds,
                 const std::vector<std::string> &Ainds, double alpha = 1.0,
                 double beta = 0.0);

    void contract(ConstTensorImplPtr A, ConstTensorImplPtr B,
                  const std::vector<std::string> &Cinds,
                  const std::vector<std::string> &Ainds,
                  const std::vector<std::string> &Binds, double alpha = 1.0,
                  double beta = 0.0);

    // => Order-2 Operations <= //

    std::map<std::string, TensorImplPtr> syev(EigenvalueOrder order) const;

    TensorImplPtr power(double alpha, double condition) const;

    void iterate(
        const std::function<void(const std::vector<size_t> &, double &)> &func);
    void citerate(const std::function<void(const std::vector<size_t> &,
                                           const double &)> &func) const;

  private:
#if defined(HAVE_ELEMENTAL)
    // => Order-2 Helper Functions <=
    void copyToElemental2(El::DistMatrix<double> &x) const;
    void copyFromElemental2(const El::DistMatrix<double> &x);
    void copyFromLowerElementalToFull2(const El::DistMatrix<double> &x);

    // => Order-1 Helper Functions <=
    void copyFromElemental1(const El::DistMatrix<double, El::VR, El::STAR> &x);
#endif

    CTF_Tensor *cyclops_;
};
}

typedef cyclops::CyclopsTensorImpl *CyclopsTensorImplPtr;
typedef const cyclops::CyclopsTensorImpl *ConstCyclopsTensorImplPtr;
}

#endif
