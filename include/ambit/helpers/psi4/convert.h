//
// Created by Justin Turney on 1/5/16.
//

#ifndef AMBIT_CONVERTER_H
#define AMBIT_CONVERTER_H

#include <libmints/mints.h>

namespace ambit {

class Tensor;

namespace helpers {

namespace psi4 {

void convert(const psi::Matrix& matrix, ambit::Tensor* target);

} // namespace psi4

} // namespace helpers

} // namespace ambit

#endif //AMBIT_CONVERTER_H
