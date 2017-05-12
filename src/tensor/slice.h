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

#if !defined(TENSOR_SLICE_H)
#define TENSOR_SLICE_H

#include "tensorimpl.h"
#include "core/core.h"
#include "disk/disk.h"

#ifdef HAVE_CYCLOPS
#include "cyclops/cyclops.h"
#endif

namespace ambit
{

void slice(TensorImplPtr C, ConstTensorImplPtr A, const IndexRange &Cinds,
           const IndexRange &Ainds, double alpha = 1.0, double beta = 0.0);

void slice(CoreTensorImplPtr C, ConstCoreTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha = 1.0,
           double beta = 0.0);

void slice(CoreTensorImplPtr C, ConstDiskTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha = 1.0,
           double beta = 0.0);

void slice(DiskTensorImplPtr C, ConstCoreTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha = 1.0,
           double beta = 0.0);

void slice(DiskTensorImplPtr C, ConstDiskTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha = 1.0,
           double beta = 0.0);

#ifdef HAVE_CYCLOPS
void slice(CoreTensorImplPtr C, ConstCyclopsTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha = 1.0,
           double beta = 0.0);

void slice(CyclopsTensorImplPtr C, ConstCoreTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha = 1.0,
           double beta = 0.0);

void slice(CyclopsTensorImplPtr C, ConstCyclopsTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha = 1.0,
           double beta = 0.0);
#endif
}

#endif
