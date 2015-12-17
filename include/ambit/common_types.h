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

static constexpr double numerical_zero__ = 1.0e-15;
}

#endif // AMBIT_COMMON_TYPES_H
