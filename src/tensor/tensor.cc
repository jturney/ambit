#include <tensor/tensor.h>
#include "tensorimpl.h"

// include header files to specific tensor types supported.
#if defined(HAVE_CYCLOPS)
#   include "cyclops/cyclops.h"
#endif

namespace tensor {

int initialize(int argc, char** argv)
{
#if defined(HAVE_CYCLOPS)
    cyclops::initialize(argc, argv);
#endif

    return 0;
}

void finialize()
{
#if defined(HAVE_CYCLOPS)
    cyclops::finalize();
#endif
}

TensorType Tensor::type() const
{
    return tensor_->type();
}

std::string Tensor::name() const
{
    return tensor_->name();
}

const std::vector<size_t>& Tensor::dims() const
{
    return tensor_->dims();
}

size_t Tensor::rank() const
{
    return tensor_->dims().size();
}

}
