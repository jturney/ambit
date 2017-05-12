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
        io::psi4::File handle(fn, io::psi4::kOpenModeOpenExisting);
        Tensor local_data =
            Tensor::build(CoreTensor, "Local Data", target.dims());
        io::psi4::IWL::read_one(handle, entry, local_data);

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
        io::psi4::IWL iwl(fn, io::psi4::kOpenModeOpenExisting);
        io::psi4::IWL::read_two(iwl, local_data);

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
