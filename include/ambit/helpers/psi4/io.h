#if !defined(TENSOR_HELPERS_PSI4_IO_H)
#define TENSOR_HELPERS_PSI4_IO_H

#include <ambit/io/io.h>

namespace ambit
{
namespace helpers
{
namespace psi4
{

/** Loads a matrix from a Psi4 data file.
 *
 * When called in an MPI run, the master node performs the read operation
 * and broadcasts the data as needed via the Tensor mechanics.
 *
 * @param fn The filename to read from.
 * @param entry The TOC entry in the Psi4 file to load.
 * @param target The target tensor to place the data.
 */
void load_matrix(const std::string &fn, const std::string &entry,
                 Tensor &target);

/** Loads two-electron integrals from an IWL file into the tensor.
*
* @param fn The filename to read from.
* @param target The tensor to place data into.
*/
void load_iwl(const std::string &fn, Tensor &target);
}
}
}

#endif
