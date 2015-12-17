#include "disk.h"
#include "memory.h"
#include "math/math.h"
#include "tensor/indices.h"
#include <sstream>
#include <string.h>
#include <cmath>
#include <unistd.h>

//#include <boost/timer/timer.hpp>

namespace ambit
{

static size_t disk_next_id__ = 0L;
size_t disk_next_id() { return disk_next_id__++; }

DiskTensorImpl::DiskTensorImpl(const string &name, const Dimension &dims)
    : TensorImpl(DiskTensor, name, dims)
{
    stringstream ss;
    ss << Tensor::scratch_path();
    ss << "/";
    ss << "DiskTensor.";
    ss << getpid();
    ss << ".";
    ss << disk_next_id();
    ss << ".dat";

    filename_ = ss.str();
    fh_ = fopen(filename_.c_str(), "wb+");
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
    for (int ind = ((int)rank()) - 1; ind >= 0; ind--)
    {
        if (fast_size * dims()[ind] <= disk_buffer__)
        {
            fast_size *= dims()[ind];
        }
        else
        {
            break;
        }
    }

    size_t slow_size = numel() / fast_size;

    double *buffer = new double[fast_size];
    memset(buffer, '\0', sizeof(double) * fast_size);

    if (beta == 0.0)
    {
        fseek(fh_, 0L, SEEK_SET);
        for (size_t ind = 0L; ind < slow_size; ind++)
        {
            fwrite(buffer, sizeof(double), fast_size, fh_);
        }
        fseek(fh_, 0L, SEEK_SET);
    }
    else
    {
        fseek(fh_, 0L, SEEK_SET);
        for (size_t ind = 0L; ind < slow_size; ind++)
        {
            fread(buffer, sizeof(double), fast_size, fh_);
            fseek(fh_, sizeof(double) * ind * fast_size, SEEK_SET);
            C_DSCAL(fast_size, beta, buffer, 1);
            fwrite(buffer, sizeof(double), fast_size, fh_);
            fseek(fh_, sizeof(double) * ind * fast_size, SEEK_SET);
        }
    }

    delete[] buffer;
}
void DiskTensorImpl::permute(ConstTensorImplPtr A, const Indices &CindsS,
                             const Indices &AindsS, double alpha, double beta)
{
    // TODO
}
}
