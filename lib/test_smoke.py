#
# @BEGIN LICENSE
#
# ambit: ambit: C++ library for the implementation of tensor product calculations
#        through a clean, concise user interface.
#
# Copyright (c) 2014-2017 Ambit developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of ambit.
#
# Ambit is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Ambit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with ambit; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import unittest
import random
import ambit
import numpy as np
import itertools as it

# Tuneage, will move later
tensor_type = ambit.TensorType.CoreTensor
max_test_rank = 4

# Sizes
dim_sizes = [9, 6, 3, 3, 3, 3, 3]
dim_inds  = 'ijklmnop'
dim_size_dict = {k:v for k, v in zip(dim_inds, dim_sizes)}

class TestOperatorOverloading(unittest.TestCase):

    def setUp(self):
        ambit.initialize()

    def tearDown(self):
        ambit.finalize()

    def build(self, name, dims, ttype=tensor_type, fill=None):
        # Builds arbitrary Tensors and numpy arrys
        # Accepts a list of integers or a string for dimensions

        if ttype != ambit.TensorType.CoreTensor:
            raise ValueError("Only CoreTensor is currently supported")

        if isinstance(dims, str):
            dims = [dim_size_dict[i] for i in dims]

        T = ambit.Tensor(ttype, name, dims)

        # Fill both N and T
        N = np.asarray(T.tensor)

        if fill:
            N.flat[:] = fill
        else:
            N.flat[:] = np.arange(np.prod(dims))

        # Copy numpy array so we no longer share memory
        N = N.copy()

        return [T, N]

    def test_2d_dot(self):
        ni, nj = 10, 20
        [aA, nA] = self.build("A", [ni, nj])
        [aB, nB] = self.build("B", [nj, ni])
        [aC, nC] = self.build("C", [ni, ni], fill=0)

        aC["ik"] = aA["ij"] * aB["jk"]


if __name__ == '__main__':
    unittest.main()


