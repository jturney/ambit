#if !defined(TENSOR_SLICE_H)
#define TENSOR_SLICE_H

#include "tensorimpl.h"
#include "core/core.h"
#include "disk/disk.h"

#ifdef HAVE_CYCLOPS
#include "cyclops/cyclops.h"
#endif

namespace ambit {

void slice(
    TensorImplPtr C,
    ConstTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha = 1.0,
    double beta = 0.0);

void slice(
    CoreTensorImplPtr C,
    ConstCoreTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha = 1.0,
    double beta = 0.0);

void slice(
    CoreTensorImplPtr C,
    ConstDiskTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha = 1.0,
    double beta = 0.0);

void slice(
    DiskTensorImplPtr C,
    ConstCoreTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha = 1.0,
    double beta = 0.0);

void slice(
    DiskTensorImplPtr C,
    ConstDiskTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha = 1.0,
    double beta = 0.0);

#ifdef HAVE_CYCLOPS
void slice(
    CoreTensorImplPtr C,
    ConstCyclopsTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha = 1.0,
    double beta = 0.0);

void slice(
    CyclopsTensorImplPtr C,
    ConstCoreTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha = 1.0,
    double beta = 0.0);

void slice(
    CyclopsTensorImplPtr C,
    ConstCyclopsTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha = 1.0,
    double beta = 0.0);
#endif

}

#endif
