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

#include <cstdlib>
#include <ambit/tensor.h>

using namespace ambit;

int main(int argc, char *argv[])
{
    ambit::initialize(argc, argv);

    {
        Tensor A = Tensor::build(CoreTensor, "A", {1000, 1000});
        Tensor B = Tensor::build(CoreTensor, "B", {1000, 1000});
        Tensor C = Tensor::build(CoreTensor, "C", {1000, 1000});

        C("ij") += A("ik") * B("jk");
    }

    ambit::finalize();

    return EXIT_SUCCESS;
}
