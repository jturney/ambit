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
