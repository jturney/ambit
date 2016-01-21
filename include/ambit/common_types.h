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
#include <functional>
#include <memory>
#include <tuple>
#include <sstream>
#include <iterator>
#include <cassert>

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
