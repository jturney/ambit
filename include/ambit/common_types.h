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
// Created by Justin Turney on 10/20/15.
//

#ifndef AMBIT_COMMON_TYPES_H
#define AMBIT_COMMON_TYPES_H

#include <cstdio>
#include <utility>
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <functional>
#include <memory>
#include <tuple>
#include <sstream>
#include <iterator>
#include <cassert>
#include <stdexcept>

namespace ambit
{

using std::tuple;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using std::string;
using std::map;
using std::pair;
using std::function;
using std::stringstream;
using std::ostringstream;
using std::ostream_iterator;
using std::istringstream;

static constexpr double numerical_zero__ = 1.0e-15;

// => Typedefs <=
using Dimension = vector<size_t>;
using IndexRange = vector<vector<size_t>>;
using Indices = vector<string>;

}

#endif // AMBIT_COMMON_TYPES_H
