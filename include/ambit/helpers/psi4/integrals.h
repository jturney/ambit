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

//
// Created by Justin Turney on 12/17/15.
//

#ifndef AMBIT_INTEGRALS_H
#define AMBIT_INTEGRALS_H

#include <libmints/mints.h>

namespace ambit
{

class Tensor;

namespace helpers
{

namespace psi4
{

void integrals(psi::OneBodyAOInt &integral, ambit::Tensor *target);

void integrals(psi::TwoBodyAOInt &integral, ambit::Tensor *target);

} // namespace psi4

} // namespace helpers

} // namespace ambit

#endif // AMBIT_INTEGRALS_H
