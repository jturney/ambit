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

from __future__ import division
from . import pyambit
import math
import copy
import numbers
import sys, traceback

class LabeledBlockedTensorProduct:

    def __init__(self, left, right):
        self.btensors = []
        self.btensors.append(left)
        self.btensors.append(right)

    def __mul__(self, other):
        if isinstance(other, LabeledBlockedTensor):
            self.btensors.append(other)
            return self

    def __float__(self):

        if len(self.btensors) != 2:
            raise RuntimeError("Conversion operator only supports binary expressions")

        # Check the indices...they must be the same
        a_indices = set(self.btensors[0].indices)
        b_indices = set(self.btensors[1].indices)

        if len(a_indices.difference(b_indices)) != 0 or len(b_indices.difference(a_indices)) != 0:
            raise RuntimeError("Non-repeated indices found in tensor dot product")

        unique_indices_key = BlockedTensor.label_to_block_keys(self.btensors[0].indices)

        index_map = {}
        k = 0
        for index in self.btensors[0].indices:
            index_map[index] = k
            k += 1

        # Setup and perform contractions
        result = 0.0
        for uik in unique_indices_key:
            prod = pyambit.LabeledTensorContraction()
            for lbt in self.btensors:
                term_key = ""
                for index in lbt.indices:
                    term_key += uik[index_map[index]]
                prod *= pyambit.ILabeledTensor(lbt.btensor.block(term_key), lbt.indices, lbt.factor)

            result += float(prod)

        return result

class LabeledBlockedTensorAddition:

    def __init__(self, left, right):
        self.tensors = []
        self.tensors.append(left)
        self.tensors.append(right)

    def __mul__(self, other):
        if isinstance(other, LabeledBlockedTensor):
            return LabeledBlockedTensorDistributive(other, self)
        elif isinstance(other, numbers.Number):
            for tensor in self.tensors:
                tensor.factor *= other
            return self

    def __neg__(self):
        for tensor in self.tensors:
            tensor.factor *= -1.0
        return self

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            for tensor in self.tensors:
                tensor.factor *= other
            return self

    def __add__(self, other):
        self.tensors.append(other)
        return self

class LabeledBlockedTensorDistributive:

    def __init__(self, left, right):
        self.A = left
        self.B = right

    def __float__(self):

        # Check the indices...they must be the same
        a_indices = set(self.A.indices)

        for B in self.B.tensors:
            b_indices = set(B.indices)

            if len(a_indices.difference(b_indices)) != 0 or len(b_indices.difference(a_indices)) != 0:
                raise RuntimeError("Non-repeated indices found in tensor dot product")

        unique_indices_key = BlockedTensor.label_to_block_keys(self.A.indices)

        index_map = {}
        k = 0
        for index in self.A.indices:
            index_map[index] = k
            k += 1

        # Setup and perform contractions
        result = 0.0
        for uik in unique_indices_key:
            prod = pyambit.LabeledTensorAddition()
            for lbt in self.B.tensors:
                term_key = ""
                for index in lbt.indices:
                    term_key += uik[index_map[index]]
                prod += pyambit.ILabeledTensor(lbt.btensor.block(term_key), lbt.indices, lbt.factor)

            term_key = ""
            for index in self.A.indices:
                term_key += uik[index_map[index]]
            A = pyambit.ILabeledTensor(self.A.btensor.block(term_key), self.A.indices, self.A.factor)
            dist = pyambit.LabeledTensorDistributive(A, prod)
            result += float(dist)

        return result

class LabeledBlockedTensor:
    def __init__(self, T, indices, factor=1.0):
        self.btensor = T
        # self.indices = indices
        # self.indices_split = pyambit.Indices.split(indices)

        self.indices = pyambit.Indices.split(indices)
        self.indices_string = indices

        self.factor = factor

    def add(self, rhs, alpha, beta):
        rhs_keys = BlockedTensor.label_to_block_keys(rhs.indices)

        perm = pyambit.Indices.permutation_order(self.indices, rhs.indices)

        for rhs_key in rhs_keys:
            lhs_key = ""
            for p in perm:
                lhs_key += rhs_key[p]

            # Grab the raw tensors
            LHS = self.btensor.block(lhs_key)
            RHS = rhs.btensor.block(rhs_key)

            # Need to protect against self assignment
            # Need to protect against different ranks
            LHS.permute(RHS, self.indices, rhs.indices, alpha * rhs.factor, beta)

    def contract(self, rhs, zero_result, add):
        if isinstance(rhs, LabeledBlockedTensorProduct):
            unique_indices = []
            for term in rhs.btensors:
                for index in term.indices:
                    unique_indices.append(index)

            unique_indices.sort()
            # print(unique_indices)

            unique_indices = list(set(unique_indices))
            unique_indices.sort()

            # print(unique_indices)

            unique_indices_key = BlockedTensor.label_to_block_keys(unique_indices)

            # print("unique_indices_key: " + str(unique_indices_key))

            index_map = {}
            k = 0
            for index in unique_indices:
                index_map[index] = k
                k += 1

            if zero_result == True:
                for uik in unique_indices_key:
                    # print("uik: " + str(uik))
                    result_key = ""
                    for index in self.indices:
                        result_key += uik[index_map[index]]
                    self.btensor.block(result_key).zero()

            # Setup and perform contractions
            for uik in unique_indices_key:
                result_key = ""
                for index in self.indices:
                    result_key += uik[index_map[index]]
                # print("result_key: " + str(result_key))
                result = pyambit.ILabeledTensor(self.btensor.block(result_key), self.indices, self.factor)

                # print("tensor: %s" % self.btensor.block(result_key).tensor)
                prod = pyambit.LabeledTensorContraction()
                for lbt in rhs.btensors:
                    term_key = ""
                    for index in lbt.indices:
                        term_key += uik[index_map[index]]
                    # print("term_key: " + str(term_key))
                    prod *= pyambit.ILabeledTensor(lbt.btensor.block(term_key), lbt.indices, lbt.factor)

                if add == True:
                    result += prod
                else:
                    result -= prod

        else:
            raise RuntimeError("LabeledBlockedTensor.contract: Unexpected type for rhs: " + type(rhs))

    def label_to_block_keys(self):
        return BlockedTensor.label_to_block_keys(self.indices)

    def __neg__(self):
        self.factor *= -1.0
        return self

    def __add__(self, other):
        if isinstance(other, LabeledBlockedTensor):
            return LabeledBlockedTensorAddition(self, other)

    def __sub__(self, other):
        other.factor *= -1.0
        return LabeledBlockedTensorAddition(self, other)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other
            return self
        elif isinstance(other, LabeledBlockedTensorAddition):
            return LabeledBlockedTensorDistributive(self, other)
        else:
            return LabeledBlockedTensorProduct(self, other)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other
            return self

    def __iadd__(self, other):
        if isinstance(other, LabeledBlockedTensor):
            self.add(other, 1.0, 1.0)
            return None
        elif isinstance(other, LabeledBlockedTensorProduct):
            self.contract(other, False, True)
            return None
        elif isinstance(other, LabeledBlockedTensorAddition):
            for tensor in other.tensors:
                self.add(tensor, 1.0, 1.0)
        elif isinstance(other, LabeledBlockedTensorDistributive):
            for tensor in other.B.tensors:
                self.contract(LabeledBlockedTensorProduct(other.A, tensor), False, True)

    def __imul__(self, other):
        if isinstance(other, numbers.Number):
            keys = self.label_to_block_keys()

            for key in keys:
                self.btensor.block(key).scale(other)

    def __isub__(self, other):
        if isinstance(other, LabeledBlockedTensor):
            self.add(other, -1.0, 1.0)
            return None
        elif isinstance(other, LabeledBlockedTensorProduct):
            self.contract(other, False, False)
            return None
        elif isinstance(other, LabeledBlockedTensorAddition):
            for tensor in other.tensors:
                self.add(tensor, -1.0, 1.0)
        elif isinstance(other, LabeledBlockedTensorDistributive):
            for tensor in other.B.tensors:
                self.contract(LabeledBlockedTensorProduct(other.A, tensor), False, False)

    def __itruediv__(self, other):
        if isinstance(other, numbers.Number):
            keys = self.label_to_block_keys()

            for key in keys:
                self.btensor.block(key).scale(1.0 / other)

    def __idiv__(self, other):
        if isinstance(other, numbers.Number):
            keys = self.label_to_block_keys()

            for key in keys:
                self.btensor.block(key).scale(1.0 / other)

