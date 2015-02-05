#include "tensorimpl.h"
#include "disk.h"
#include "memory.h"
#include "math/math.h"
#include "indices.h"
#include <string.h>
#include <cmath>

#include <boost/timer/timer.hpp>

namespace tensor {

DiskTensorImpl::DiskTensorImpl(const std::string& name, const Dimension& dims)
        : TensorImpl(kDisk, name, dims)
{
    filename_ = "dummy.dat"; // TODO
    fh_ = fopen(filename_.c_str(),"wb");  
    scale(0.0); // Prestripe
}
DiskTensorImpl::~DiskTensorImpl()
{
    fclose(fh_);
    remove(filename_.c_str());
}
void DiskTensorImpl::scale(double beta)
{
    size_t fast_size = 1L;
    if (rank() > 0) fast_size *= dims()[rank()-1];
    if (rank() > 1) fast_size *= dims()[rank()-2];

    size_t slow_size = numel() / fast_size;

    double* buffer = new double[fast_size];
    memset(buffer,'\0',sizeof(double)*fast_size);

    if (beta == 0.0) {
        fseek(fh_,0L,SEEK_SET);
        for (size_t ind = 0L; ind < slow_size; ind++) {
            fwrite(buffer, sizeof(double), fast_size, fh_);
        }
        fseek(fh_,0L,SEEK_SET);
    } else {
        fseek(fh_,0L,SEEK_SET);
        for (size_t ind = 0L; ind < slow_size; ind++) {
            fread(buffer, sizeof(double), fast_size, fh_);
            fseek(fh_,sizeof(double)*ind*fast_size,SEEK_SET);
            C_DSCAL(fast_size,beta,buffer,1);
            fwrite(buffer, sizeof(double), fast_size, fh_);
            fseek(fh_,sizeof(double)*ind*fast_size,SEEK_SET);
        }
    }

    delete[] buffer;    

}
void DiskTensorImpl::permute(
    ConstTensorImplPtr A,
    const Indices& CindsS,
    const Indices& AindsS,
    double alpha,
    double beta)
{
    // TODO
}
void DiskTensorImpl::slice(
    ConstTensorImplPtr A,
    const IndexRange& Cinds,
    const IndexRange& Ainds,
    double alpha,
    double beta)
{
    // TODO
}

}
