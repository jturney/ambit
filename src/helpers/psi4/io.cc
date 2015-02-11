#include <tensor/tensor.h>
#include <tensor/helpers/psi4/io.h>

namespace tensor {
namespace helpers {
namespace psi4 {

/** Loads a matrix from a Psi4 data file.
 *
 * When called in an MPI run, the master node performs the read operation
 * and broadcasts the data as needed via the Tensor mechanics.
 *
 * @param fn The filename to read from.
 * @param entry The TOC entry in the Psi4 file to load.
 * @param target The target tensor to place the data.
 */
void load_matrix(const std::string& fn, const std::string& entry, Tensor& target)
{
    std::vector<double> data;

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

}
}
}
