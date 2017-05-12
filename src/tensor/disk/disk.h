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

#if !defined(TENSOR_DISK_H)
#define TENSOR_DISK_H

#include "tensor/tensorimpl.h"

namespace ambit
{

/// 1 GiB in doubles
static constexpr size_t disk_buffer__ = 125000000L;

class DiskTensorImpl : public TensorImpl
{
  public:
    DiskTensorImpl(const std::string &name, const Dimension &dims);
    ~DiskTensorImpl();

    void scale(double beta = 0.0);

    void permute(ConstTensorImplPtr A, const std::vector<std::string> &Cinds,
                 const std::vector<std::string> &Ainds, double alpha = 1.0,
                 double beta = 0.0);

    std::string filename() const { return filename_; }
    FILE *fh() const { return fh_; }

  private:
    std::string filename_;
    FILE *fh_;
};

typedef DiskTensorImpl *DiskTensorImplPtr;
typedef const DiskTensorImpl *ConstDiskTensorImplPtr;
}

#endif
