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

print("I think I can make it this far...")
import unittest
print("But not this far...")
import random
import ambit
import numpy as np
import itertools as it

# Tuneage, will move later
tensor_type = ambit.TensorType.CoreTensor
max_test_rank = 4

class TestOperatorOverloading(unittest.TestCase):

    def setUp(self):
        ambit.initialize()

    def tearDown(self):
        ambit.finalize()


    def test_2d_dot(self):

        ni, nj = 10, 20


if __name__ == '__main__':
    unittest.main()


