from __future__ import division
from . import pyambit
import numbers
import itertools

class LabeledTensorProduct:
    def __init__(self, left, right):
        self.tensors = []
        if left: self.tensors.append(left)
        if right: self.tensors.append(right)

    def __mul__(self, other):
        if isinstance(other, LabeledTensor):
            self.tensors.append(other)
            return self
        else:
            print("LabeledTensorProduct::mul %s not implemented" % (type(other)))
            return NotImplemented

    def __float__(self):
        if len(self.tensors) != 2:
            raise RuntimeError("Conversion operator only supports binary expressions.")

        R = Tensor(self.tensors[0].tensor.type, "R", [])
        R.contract(self.tensors[0], self.tensors[1], [], self.tensors[0].indices, self.tensors[1].indices,
                   self.tensors[0].factor * self.tensors[1].factor, 1.0)

        C = Tensor(pyambit.TensorType.kCore, "C", [])
        C.slice(R, [], [])

        return C.data()[0]

    def compute_contraction_cost(self, perm):
        indices_to_size = {}

        for ti in self.tensors:
            for i, v in enumerate(ti.indices):
                indices_to_size[v] = ti.tensor.dims[i]

        # print("indices_to_size: " + str(indices_to_size))

        cpu_cost_total = 0.0
        memory_cost_max = 0.0

        first = self.tensors[perm[0]].indices

        for i in perm[1:]:
            first = set(first)
            second = set(self.tensors[i].indices)

            common = first.intersection(second)
            first_unique = first.difference(second)
            second_unique = second.difference(first)

            common_size = 1.0
            for s in common: common_size *= indices_to_size[s]
            first_size = 1.0
            for s in first: first_size *= indices_to_size[s]
            second_size = 1.0
            for s in second: second_size *= indices_to_size[s]
            first_unique_size = 1.0
            for s in first_unique: first_unique_size *= indices_to_size[s]
            second_unique_size = 1.0
            for s in second_unique: second_unique_size *= indices_to_size[s]
            result_size = first_unique_size + second_unique_size

            stored_indices = []
            for v in first_unique: stored_indices.append(v)
            for v in second_unique: stored_indices.append(v)

            cpu_cost = common_size * result_size
            memory_cost = first_size + second_size + result_size
            cpu_cost_total += cpu_cost
            memory_cost_max += max(memory_cost_max, memory_cost)

            first = stored_indices

        return [cpu_cost_total, memory_cost_max]

class LabeledTensorAddition:
    def __init__(self, left, right):
        self.tensors = []
        if left: self.tensors.append(left)
        if right: self.tensors.append(right)

    def __mul__(self, other):
        if isinstance(other, LabeledTensor):
            return LabeledTensorDistributive(other, self)
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


class LabeledTensorDistributive:
    def __init__(self, left, right):
        self.A = left
        self.B = right

    def __float__(self):
        R = Tensor(self.A.tensor.type, "R", [])

        for tensor in self.B.tensors:
            R.contract(self.A, tensor, [], self.A.indices, tensor.indices, self.A.factor * tensor.factor, 1.0)

        C = Tensor(pyambit.TensorType.kCore, "C", [])
        C.slice(R, [], [])

        return C.data()[0]


class LabeledTensor:
    def __init__(self, t, indices, factor=1.0):
        self.factor = factor
        self.tensor = t
        if isinstance(indices, pyambit.Indices):
            self.indices = indices
        else:
            self.indices = pyambit.Indices.split(indices)

    def dim_by_index(self, index):
        positions = [i for i,x in enumerate(self.indices) if x == index]
        if len(positions) != 1:
            raise RuntimeError("LabeledTensor.dim_by_index: Couldn't find index " + index)
        return self.tensor.dims[positions[0]]

    def __neg__(self):
        self.factor *= -1.0
        return self

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other
            return self
        elif isinstance(other, LabeledTensorAddition):
            return LabeledTensorDistributive(self, other)
        else:
            return LabeledTensorProduct(self, other)

    def __add__(self, other):
        return LabeledTensorAddition(self, other)

    def __sub__(self, other):
        other.factor *= -1.0
        return LabeledTensorAddition(self, other)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other

            return self
        else:
            return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, LabeledTensor):
            self.tensor.permute(other.tensor, self.indices, other.indices, other.factor, self.factor)
            return None
        elif isinstance(other, LabeledTensorDistributive):
            raise NotImplementedError("LabeledTensor.__iadd__(LabeledTensorDistributive) is not implemented")
        elif isinstance(other, LabeledTensorProduct):
            nterms = len(other.tensors)
            best_perm = [0 for x in range(nterms)]
            perms = [x for x in range(nterms)]
            best_cpu_cost = 1.0e200
            best_memory_cost = 1.0e200

            for perm in itertools.permutations(perms):
                [cpu_cost, memory_cost] = other.compute_contraction_cost(perm)
                if cpu_cost < best_cpu_cost:
                    best_perm = perm
                    best_cpu_cost = cpu_cost
                    best_memory_cost = memory_cost

            # At this point best_perm should be used to perform the contractions in optimal order
            A = other.tensors[best_perm[0]]
            maxn = nterms - 2
            for n in range(maxn):
                B = other.tensors[best_perm[n + 1]]

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

                tAB = Tensor.build(A.tensor.type, A.tensor.name + " * " + B.tensor.name, dims)

                tAB.contract(A, B, indices, A.indices, B.indices, A.factor * B.factor, 0.0)

                A.set(LabeledTensor(tAB.tensor, "".join(indices), 1.0))

            B = other.tensors[best_perm[nterms - 1]]

            self.tensor.contract(A.tensor, B.tensor, self.indices, A.indices, B.indices, A.factor * B.factor,
                                 self.factor)

            # This operator is complete.
            return None

        elif isinstance(other, LabeledTensorAddition):
            raise NotImplementedError("LabeledTensor.__iadd__(LabeledTensorAddition) is not implemented")
        else:
            raise NotImplementedError("LabeledTensor.__iadd__(%s) is not implemented" % (type(other)))

    def __isub__(self, other):
        if isinstance(other, LabeledTensor):
            self.tensor.permute(other.tensor, self.indices, other.indices, -other.factor, self.factor)
            return None
        elif isinstance(other, LabeledTensorDistributive):
            raise NotImplementedError("LabeledTensor.__isub__(%s) is not implemented" % (type(other)))
        elif isinstance(other, LabeledTensorProduct):
            nterms = len(other.tensors)
            best_perm = [0 for x in range(nterms)]
            perms = [x for x in range(nterms)]
            best_cpu_cost = 1.0e200
            best_memory_cost = 1.0e200

            for perm in itertools.permutations(perms):
                [cpu_cost, memory_cost] = other.compute_contraction_cost(perm)
                if cpu_cost < best_cpu_cost:
                    best_perm = perm
                    best_cpu_cost = cpu_cost
                    best_memory_cost = memory_cost

            # At this point best_perm should be used to perform the contractions in optimal order
            A = other.tensors[best_perm[0]]
            maxn = nterms - 2
            for n in range(maxn):
                B = other.tensors[best_perm[n + 1]]

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

                tAB = Tensor.build(A.tensor.type, A.tensor.name + " * " + B.tensor.name, dims)

                tAB.contract(A, B, indices, A.indices, B.indices, A.factor * B.factor, 0.0)

                A.set(LabeledTensor(tAB.tensor, "".join(indices), 1.0))

            B = other.tensors[best_perm[nterms - 1]]

            self.tensor.contract(A.tensor, B.tensor, self.indices, A.indices, B.indices, -A.factor * B.factor,
                                 self.factor)

            # This operator is complete.
            return None

        elif isinstance(other, LabeledTensorAddition):
            raise NotImplementedError("LabeledTensor.__isub__(%s) is not implemented" % (type(other)))
        else:
            print("LabeledTensor::__isub__ not implemented for this type.")
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, numbers.Number):
            self.tensor.scale(other)
            return None
        else:
            raise NotImplementedError("LabeledTensor.__isub__(%s) is not implemented" % (type(other)))

    def __itruediv__(self, other):
        if isinstance(other, numbers.Number):
            self.tensor.scale(1.0 / other)
            return None
        else:
            raise NotImplementedError("LabeledTensor.__isub__(%s) is not implemented" % (type(other)))

    def __idiv__(self, other):
        if isinstance(other, numbers.Number):
            self.tensor.scale(1.0 / other)
            return None
        else:
            raise NotImplementedError("LabeledTensor.__isub__(%s) is not implemented" % (type(other)))

    def set(self, to):
        self.tensor = to.tensor
        self.indices = to.indices
        self.factor = to.factor


class Tensor:
    @staticmethod
    def build(type, name, dims):
        return Tensor(type, name, dims)

    def __init__(self, type=None, name=None, dims=None, existing=None):
        if existing:
            self.tensor = existing
            self.type = existing.type
            self.dims = existing.dims
            self.name = name if name else existing.name
        else:
            self.name = name
            self.rank = len(dims)
            self.type = type
            self.dims = dims
            self.tensor = pyambit.ITensor.build(type, name, dims)

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return SlicedTensor(self, indices)
        else:
            return LabeledTensor(self.tensor, indices)

    def __setitem__(self, indices_str, value):

        if isinstance(value, SlicedTensor):
            if self.tensor == value.tensor:
                raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise RuntimeError("SlicedTensor::__setitem__: Sliced tensors do not have same rank")

            self.tensor.slice(value.tensor.tensor, indices_str, value.range, value.factor, 0.0)

            return None

        indices = pyambit.Indices.split(str(indices_str))

        if isinstance(value, LabeledTensorProduct):
            nterms = len(value.tensors)
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
            A = value.tensors[best_perm[0]]
            maxn = nterms - 2
            for n in range(maxn):
                B = value.tensors[best_perm[n + 1]]

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

                tAB = Tensor.build(A.tensor.type, A.tensor.name + " * " + B.tensor.name, dims)

                tAB.contract(A, B, indices, A.indices, B.indices, A.factor * B.factor, 0.0)

                A.set(LabeledTensor(tAB.tensor, "".join(indices), 1.0))

            B = value.tensors[best_perm[nterms - 1]]

            self.tensor.contract(A.tensor, B.tensor, indices, A.indices, B.indices, A.factor * B.factor,
                          0.0)

            # This operator is complete.
            return None


        elif isinstance(value, LabeledTensorAddition):
            self.tensor.zero()

            for tensor in value.tensors:
                if isinstance(tensor, LabeledTensor):
                    self.tensor.permute(tensor.tensor, indices, tensor.indices, tensor.factor, 1.0)
                else:
                    # recursively call set item
                    self.factor = 1.0
                    self.__setitem__(indices_str, tensor)

        # This should be handled by LabeledTensor above
        elif isinstance(value, LabeledTensor):

            if self == value.tensor:
                raise RuntimeError("Self-assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise ArithmeticError("Permuted tensors do not have same rank")

            self.tensor.permute(value.tensor, indices, value.indices, value.factor, 0.0)

        elif isinstance(value, LabeledTensorDistributive):

            self.tensor.zero()

            A = value.A
            for B in value.B.tensors:
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

    def data(self):
        return self.tensor.data()

    def norm(self, type):
        return self.tensor.norm(type)

    def zero(self):
        self.tensor.zero()

    def scale(self, beta):
        self.tensor.scale(beta)

    def copy(self, other):
        self.tensor.copy(other)

    def slice(self, A, Cinds, Ainds, alpha=1.0, beta=0.0):
        self.tensor.slice(A.tensor, Cinds, Ainds, alpha, beta)

    def permute(self, A, Cinds, Ainds, alpha=1.0, beta=0.0):
        self.tensor.permute(A.tensor, Cinds, Ainds, alpha, beta)

    def contract(self, A, B, Cinds, Ainds, Binds, alpha=1.0, beta=0.0):
        self.tensor.contract(A.tensor, B.tensor, Cinds, Ainds, Binds, alpha, beta)

    def gemm(self, A, B, transA, transB, nrow, ncol, nzip, ldaA, ldaB, ldaC, offA=0, offB=0, offC=0, alpha=1.0,
             beta=0.0):
        self.tensor.gemm(A.tensor, B.tensor, transA, transB, nrow, ncol, nzip, ldaA, ldaB, ldaC, offA, offB, offC,
                         alpha, beta)

    def syev(self, order):
        aResults = self.tensor.syev(order)

        results = {}
        for k, v in aResults.iteritems():
            results[k] = Tensor(existing=v)

        return results

    def power(self, p, condition=1.0e-12):
        aResult = self.tensor.power(p, condition)
        return Tensor(existing=aResult)


class SlicedTensor:
    def __init__(self, tensor, range, factor=1.0):
        self.tensor = tensor
        self.range = range
        self.factor = factor

        # Check the data given to us
        if not isinstance(tensor, Tensor):
            raise RuntimeError("SlicedTensor: Expected tensor to be Tensor")

        if len(range) != tensor.rank:
            raise RuntimeError(
                "SlicedTensor: Sliced tensor does not have correct number of indices for underlying tensor's rank")

        for idx, value in enumerate(range):
            if len(value) != 2:
                raise RuntimeError(
                    "SlicedTensor: Each index of an IndexRange should have two elements {start,end+1} in it.")
            if value[0] > value[1]:
                raise RuntimeError("SlicedTensor: Each index of an IndexRange should end+1>=start in it.")
            if value[1] > tensor.dims[idx]:
                raise RuntimeError("SlicedTensor: IndexRange exceeds size of tensor.")

    def __iadd__(self, value):
        if isinstance(value, SlicedTensor):
            if self.tensor == value.tensor:
                raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise RuntimeError("SlicedTensor::__setitem__: Sliced tensors do not have same rank")

            self.tensor.slice(value.tensor, self.range, value.range, value.factor, 1.0)

            return None
        else:
            raise NotImplementedError("SlicedTensor.__iadd__(%s) is not implemented" % (type(other)))


    def __isub__(self, value):
        if isinstance(value, SlicedTensor):
            if self.tensor == value.tensor:
                raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise RuntimeError("SlicedTensor::__setitem__: Sliced tensors do not have same rank")

            self.tensor.slice(value.tensor, self.range, value.range, -value.factor, 1.0)

            return None
        else:
            raise NotImplementedError("SlicedTensor.__isub__(%s) is not implemented" % (type(other)))


    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return SlicedTensor(self.tensor, self.range, other * self.factor)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return SlicedTensor(self.tensor, self.range, other * self.factor)

    def __neg__(self):
        return SlicedTensor(self.tensor, self.range, -self.factor)
