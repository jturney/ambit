#include <ambit/tensor.h>
#include <ambit/helpers/psi4/io.h>
#include <ambit/timer.h>

namespace ambit
{
namespace helpers
{
namespace psi4
{

void load_matrix(const std::string &fn, const std::string &entry,
                 Tensor &target)
{
    timer::timer_push("ambit::helpers::psi4::load_matrix");
    if (settings::rank == 0)
    {
        io::File handle(fn, io::kOpenModeOpenExisting);
        Tensor local_data = Tensor::build(CoreTensor, "Local Data", target.dims());
        io::IWL::read_one(handle, entry, local_data);

        target() = local_data();
    }
    else
    {
        Dimension zero;
        IndexRange zero_range;

        for (size_t i = 0; i < target.rank(); ++i)
        {
            zero.push_back(0);
            zero_range.push_back({0, 0});
        }
        Tensor local_data = Tensor::build(CoreTensor, "Local Data", zero);

        target(zero_range) = local_data(zero_range);
    }
    timer::timer_pop();
}

void load_iwl(const std::string &fn, Tensor &target)
{
    timer::timer_push("ambit::helpers::psi4::load_iwl");
    if (settings::rank == 0)
    {
        Tensor local_data = Tensor::build(CoreTensor, "g", target.dims());
        io::IWL iwl(fn, ambit::io::kOpenModeOpenExisting);
        io::IWL::read_two(iwl, local_data);

        target() = local_data();
    }
    else
    {
        Dimension zero;
        IndexRange zero_range;

        for (size_t i = 0; i < target.rank(); ++i)
        {
            zero.push_back(0);
            zero_range.push_back({0, 0});
        }
        Tensor local_data = Tensor::build(CoreTensor, "Local Data", zero);

        target(zero_range) = local_data(zero_range);
    }
    timer::timer_pop();
}
}
}
}
