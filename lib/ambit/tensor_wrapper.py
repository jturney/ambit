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
import numbers
import itertools
import numpy as np

def _parse_slices(indices, dims):
    # Multiarray slices
    if isinstance(indices, (list, tuple)):

        # 1D Slice
        if isinstance(indices[0], int):
            return list(indices)

        # Throw if dims do not match
        if len(indices) != len(dims):
            raise RuntimeError("SlicedTensor: Number of slices does not equal tensor rank.")

        # ND slice
        formed_indices = []
        for num, sl in enumerate(indices):
            if isinstance(sl, list):
                formed_indices.append(sl)
            elif isinstance(sl, slice):
                if sl.step:
                    raise ValueError("Step slices are not supported.")
                new_slice = [sl.start if sl.start else 0, sl.stop if sl.stop else dims[num]]
                formed_indices.append(new_slice)
            else:
                raise ValueError("Slice of type %s is not supported." % type(sl))
        return formed_indices

    # Single slice
    elif isinstance(indices, slice):
        if indices.step:
            raise ValueError("Step slices are not supported.")
        sl = indices
        return [[sl.start if sl.start else 0, sl.stop if sl.stop else dims[num]]]

    # Conventional list slices
    else:
        return indices


class Tensor:
    @staticmethod
    def build(dtype, name, dims):
        """
        Factory constructor. Builds a Tensor of TensorType type
        with given name and dimensions dims.

        :param type: the TensorType to build
        :param name: the name of the Tensor
        :param dims: the dimensions of the indices of the tensor
                     (len(dims) is the tensor rank
        :return: new Tensor of TensorType type with name and dims.
                 The returned Tensor is set to zero.
        """
        return Tensor(dtype, name, dims)

    def __init__(self, dtype=None, name=None, dims=None, existing=None):

        if isinstance(dtype, str):
            dtype = pyambit.TensorType.names[dtype]

        if existing:
            self.tensor = existing
            self.dtype = existing.dtype
            self.dims = existing.dims
            self.name = name if name else existing.name

        else:
            self.name = name
            self.rank = len(dims)
            self.dtype = dtype
            self.dims = dims
            self.tensor = pyambit.ITensor.build(dtype, name, dims)

    @property
    def __array_interface__(self):
        if self.dtype == pyambit.TensorType.CoreTensor:
            return self.tensor.__array_interface__
        else:
            raise TypeError('Only CoreTensor tensors can be converted to ndarrays.')

    def __getitem__(self, indices):

        if isinstance(indices, str):
            return pyambit.ILabeledTensor(self.tensor, indices)
        else:
            return pyambit.SlicedTensor(self.tensor, _parse_slices(indices, self.dims))

    def __setitem__(self, indices_str, value):
        if isinstance(value, pyambit.SlicedTensor):
            if self.tensor == value.tensor:
                raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise RuntimeError(f"SlicedTensor::__setitem__: Sliced tensors do not have same rank, {self.tensor.rank} vs. {value.tensor.rank}")
            if isinstance(indices_str, str):
                raise TypeError("SlicedTensor::__setitem__: Slice cannot be a string")

            indices = _parse_slices(indices_str, self.dims)
            self.tensor.slice(value.tensor, indices, value.range, value.factor, 0.0)

            return None

        indices = pyambit.Indices.split(str(indices_str))

        if isinstance(value, pyambit.LabeledTensorContraction):
            nterms = len(value)
            best_perm = [0 for x in range(nterms)]
            perms = [x for x in range(nterms)]
            best_cpu_cost = 1.0e200
            best_memory_cost = 1.0e200

            for perm in itertools.permutations(perms):
                [cpu_cost, memory_cost] = value.compute_contraction_cost(perm)
                if cpu_cost < best_cpu_cost:
                    best_perm = perm
                    best_cpu_cost = cpu_cost
                    best_memory_cost = memory_cost

            # At this point best_perm should be used to perform the contractions in optimal order
            A = value[best_perm[0]]
            maxn = nterms - 2
            for n in range(maxn):
                B = value[best_perm[n + 1]]

                AB_indices = pyambit.Indices.determine_contraction_result_from_indices(A.indices, B.indices)
                A_fix_idx = AB_indices[1]
                B_fix_idx = AB_indices[2]

                dims = []
                indices = []

                for index in A_fix_idx:
                    dims.append(A.dim_by_index(index))
                    indices.append(index)
                for index in B_fix_idx:
                    dims.append(B.dim_by_index(index))
                    indices.append(index)

                tAB = Tensor.build(A.tensor.dtype, A.tensor.name + " * " + B.tensor.name, dims)

                tAB.contract(A, B, indices, A.indices, B.indices, A.factor * B.factor, 0.0)

                A.set(LabeledTensor(tAB.tensor, "".join(indices), 1.0))

            B = value[best_perm[nterms - 1]]

            self.tensor.contract(A.tensor, B.tensor, indices, A.indices, B.indices, A.factor * B.factor,
                                 0.0)

            # This operator is complete.
            return None


        elif isinstance(value, pyambit.LabeledTensorAddition):
            self.tensor.zero()

            for tensor in value:
                if isinstance(tensor, pyambit.ILabeledTensor):
                    self.tensor.permute(tensor.tensor, indices, tensor.indices, tensor.factor, 1.0)
                else:
                    # recursively call set item
                    self.factor = 1.0
                    self.__setitem__(indices_str, tensor)

        # This should be handled by LabeledTensor above
        elif isinstance(value, pyambit.ILabeledTensor):

            if self == value.tensor:
                raise RuntimeError("Self-assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise ArithmeticError("Permuted tensors do not have same rank")

            self.tensor.permute(value.tensor, indices, value.indices, value.factor, 0.0)

        elif isinstance(value, pyambit.LabeledTensorDistributive):

            self.tensor.zero()

            A = value.A
            for B in value.B:
                self.tensor.contract(A.tensor, B.tensor, indices, A.indices, B.indices, A.factor * B.factor, 1.0)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.tensor == other.tensor
        elif isinstance(other, pyambit.ITensor):
            return self.tensor == other
        else:
            raise NotImplementedError("LabeledTensor.__eq__(%s) is not implemented" % (type(other)))

    def printf(self):
        self.tensor.printf()

    def set(self, alpha):
        """
        Set all elements of the tensor to alpha.
        """
        self.tensor.set(alpha)

    def norm(self, type):
        """
        Returns the norm of the tensor.

        :param type: the type of norm desired:
         0 - Infinity-norm, maximum absolute value of elements
         1 - One-norm, sum of absolute values of elements
         2 - Two-norm, square root of sum of squares

        :return: computed norm
        """
        return self.tensor.norm(type)

    def zero(self):
        """
        Sets the data of the tensor to zeros.
        """
        self.tensor.zero()

    def scale(self, beta):
        """
        Scales the tensor by scalar beta, e.g.
        C = beta * C

        Note: If beta is 0.0, a memset is performed rather than a scale to clamp
        NaNs and other garbage out.

        :param beta: the scale to apply
        """
        self.tensor.scale(beta)

    def copy(self, other):
        """
        Copy the data of other into this tensor:
        C() = other()

        :param other: the tensor to copy data from
        """
        self.tensor.copy(other)

    def slice(self, A, Cinds, Ainds, alpha=1.0, beta=0.0):
        """
        Perform the slice:
         C(Cinds) = alpha * A(Ainds) + beta * C(Cinds)

        Note: Most users should instead use the operator overloading
        routines, e.g.,
         C2[[[0,m],[0,n]]] += 0.5 * A2[[[1,m+1],[1,n+1]]]

        :param A: the source tensor, e.g. A2
        :param Cinds: the slices of indices of tensor C, e.g., [[0,m],[0,n]]
        :param Ainds: the indices of tensor A, e.g., [[1,m+1],[1,n+1]]
        :param alpha: the scale applied to the tensor A, e.g. 0.5
        :param beta: the scale applied to the tensor C, e.g., 1.0

        Results:
         C is the current tensor, whose data is overwritten. e.g., C2
         All elements outside of the IndexRange in C are untouched, alpha and beta
         scales are applied only to elements indices of the IndexRange
        """
        self.tensor.slice(A.tensor, Cinds, Ainds, alpha, beta)

    def permute(self, A, Cinds, Ainds, alpha=1.0, beta=0.0):
        """
        Perform the permutation:
        C[Cinds] = alpha * A[Ainds] + beta * C[Cinds]

        Note: Most users should instead use the operator overloading
        routines, e.g.,
        C["ij"] += 0.5 * A2["ji"]

        :param A: the source tensor, e.g., A2
        :param Cinds: the indices of tensor C, e.g., "ij"
        :param Ainds: the indices of tensor A, e.g., "ji"
        :param alpha: the scale applied to the tensor A, e.g., 0.5
        :param beta: the scale applied to the tensor C, e.g., 1.0
        """
        self.tensor.permute(A.tensor, Cinds, Ainds, alpha, beta)

    def min(self):
        return self.tensor.min()

    def max(self):
        return self.tensor.max()

    def contract(self, A, B, Cinds, Ainds, Binds, alpha=1.0, beta=0.0):
        """
        Perform the contraction:
         C[Cinds] = alpha * A[Ainds] * B[Binds] + beta * C[Cinds]

        Note: Most users should instead use the operator overloading
        routines, e.g.,
         C2["ij"] += 0.5 * A2["ik"] * B2["jk"]

        :param A: the left-side factor tensor, e.g., A2
        :param B: the right-side factor tensor, e.g., B2
        :param Cinds: the indices of tensor C, e.g., "ij"
        :param Ainds: the indices of tensor A, e.g., "ik"
        :param Binds: the indices of tensor B, e.g., "jk"
        :param alpha: the scale applied to the product A*B, e.g., 0.5
        :param beta: the scale applied to the tensor C, e.g., 1.0
        """
        self.tensor.contract(A.tensor, B.tensor, Cinds, Ainds, Binds, alpha, beta)

    def gemm(self, A, B, transA, transB, nrow, ncol, nzip, ldaA, ldaB, ldaC, offA=0, offB=0, offC=0, alpha=1.0,
             beta=0.0):
        """
        Perform the GEMM call equivalent to:
         C_DGEMM(
             (transA ? 'T' : 'N'),
             (transB ? 'T' : 'N'),
             nrow,
             ncol,
             nzip,
             alpha,
             Ap + offA,
             ldaA,
             Bp + offB,
             ldaB,
             beta,
             Cp + offC,
             ldaC);
         where, e.g., Ap = A.data().data();

        Notes:
         - This is only implemented for CoreTensor
         - No bounds checking on the GEMM is performed
         - This function is intended to help advanced users get optimal
           performance from single-node codes.

        :param A: the left-side factor tensor
        :param B: the right-side factor tensor
        :param transA: transpose A or not
        :param transB: transpose B or not
        :param nrow: number of rows in the GEMM call
        :param ncol: number of columns in the GEMM call
        :param nzip: number of zip indices in the GEMM call
        :param ldaA: leading dimension of A:
                     Must be >= nzip if transA == False
                     Must be >= nrow if transA == True
        :param ldaB: leading dimension of B:
                     Must be >= ncol if transB == False
                     Must be >= nzip if transB == True
        :param ldaC: leading dimension of C:
                     Must be >= ncol
        :param offA: the offset of the A data pointer to apply
        :param offB: the offset of the B data pointer to apply
        :param offC: the offset of the C data pointer to apply
        :param alpha: the scale to apply to A*B
        :param beta: the scale to apply to C
        """
        self.tensor.gemm(A.tensor, B.tensor, transA, transB, nrow, ncol, nzip, ldaA, ldaB, ldaC, offA=0, offB=0, offC=0,
                         alpha=1.0, beta=0.0)

    def syev(self, order=None):

        if order is None:
            order = pyambit.EigenvalueOrder.names["AscendingEigenvalue"]
        elif isinstance(order, str):
            order = pyambit.EigenvalueOrder.names[order]
        elif isinstance(order, pyambit.EigenvalueOrder):
            order = order
        else:
            raise ValueError("Tensor: syev order type %s not recognized", type(order))

        aResults = self.tensor.syev(order)

        results = {}
        for k in ['eigenvectors', 'eigenvalues']:
            results[k] = Tensor(existing=aResults[k])

        return results

    def power(self, p, condition=1.0e-12):
        aResult = self.tensor.power(p, condition)
        return Tensor(existing=aResult)

