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

#include <random>

#include <ambit/blocked_tensor.h>

using namespace ambit;

int main(int argc, char *argv[])
{
    ambit::initialize(argc, argv);

    {
        BlockedTensor::add_mo_space("o", "i,j,k,l", {0, 1, 2, 3, 4}, AlphaSpin);
        BlockedTensor::add_mo_space("v", "a,b,c,d", {5, 6, 7, 8, 9}, AlphaSpin);
        BlockedTensor::add_mo_space("O", "I,J,K,L", {0, 1, 2, 3, 4}, BetaSpin);
        BlockedTensor::add_mo_space("V", "A,B,C,D", {5, 6, 7, 8, 9}, BetaSpin);

        BlockedTensor::print_mo_spaces();

        BlockedTensor F =
            BlockedTensor::build(CoreTensor, "F", {"oo", "ov", "vo"});

        F.iterate([](const std::vector<size_t> & /*indices*/,
                     const std::vector<SpinType> & /*spin*/, double &value)
                  {
                      value = double(std::rand()) / double(RAND_MAX);
                  });

        F.print(stdout);
    }

    ambit::finalize();

    return EXIT_SUCCESS;
}
