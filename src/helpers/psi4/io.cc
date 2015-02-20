#include <tensor/tensor.h>
#include <tensor/helpers/psi4/io.h>

namespace tensor {
namespace helpers {
namespace psi4 {

void load_matrix(const std::string& fn, const std::string& entry, Tensor& target)
{
    if (settings::rank == 0) {
        io::File handle(fn, io::kOpenModeOpenExisting);
        Tensor local_data = Tensor::build(kCore, "Local Data", target.dims());
        io::IWL::read_one(handle, entry, local_data);

        target() = local_data();
    }
    else {
        Dimension zero;
        IndexRange zero_range;

        for (size_t i=0; i<target.rank(); ++i) {
            zero.push_back(0);
            zero_range.push_back({0, 0});
        }
        Tensor local_data = Tensor::build(kCore, "Local Data", zero);

        target(zero_range) = local_data(zero_range);
    }
}

void load_iwl(const std::string& fn, Tensor& target)
{
    if (settings::rank == 0) {
        Tensor local_data = Tensor::build(kCore, "g", target.dims());
        io::IWL iwl(fn, tensor::io::kOpenModeOpenExisting);
        io::IWL::read_two(iwl, local_data);

        target() = local_data();
    }
    else {
        Dimension zero;
        IndexRange zero_range;

        for (size_t i=0; i<target.rank(); ++i) {
            zero.push_back(0);
            zero_range.push_back({0, 0});
        }
        Tensor local_data = Tensor::build(kCore, "Local Data", zero);

        target(zero_range) = local_data(zero_range);
    }
}

}
}
}
