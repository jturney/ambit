from __future__ import division
from . import pyambit
import numbers
import itertools


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


class LabeledTensorProduct:
    def __init__(self, left, right):
        self.tensors = []
        if left:
            self.tensors.append(left)
        if right:
            self.tensors.append(right)

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

        R = Tensor(self.tensors[0].tensor.dtype, "R", [])
        R.contract(self.tensors[0], self.tensors[1], [], self.tensors[0].indices, self.tensors[1].indices,
                   self.tensors[0].factor * self.tensors[1].factor, 1.0)

        C = Tensor(pyambit.TensorType.CoreTensor, "C", [])
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
            for s in common:
                common_size *= indices_to_size[s]
            first_size = 1.0
            for s in first:
                first_size *= indices_to_size[s]
            second_size = 1.0
            for s in second:
                second_size *= indices_to_size[s]
            first_unique_size = 1.0
            for s in first_unique:
                first_unique_size *= indices_to_size[s]
            second_unique_size = 1.0
            for s in second_unique:
                second_unique_size *= indices_to_size[s]
            result_size = first_unique_size + second_unique_size

            stored_indices = []
            for v in first_unique:
                stored_indices.append(v)
            for v in second_unique:
                stored_indices.append(v)

            cpu_cost = common_size * result_size
            memory_cost = first_size + second_size + result_size
            cpu_cost_total += cpu_cost
            memory_cost_max += max(memory_cost_max, memory_cost)

            first = stored_indices

        return [cpu_cost_total, memory_cost_max]


class LabeledTensorAddition:
    def __init__(self, left, right):
        self.tensors = []
        if left:
            self.tensors.append(left)
        if right:
            self.tensors.append(right)

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
        R = Tensor(self.A.tensor.dtype, "R", [])

        for tensor in self.B.tensors:
            R.contract(self.A, tensor, [], self.A.indices, tensor.indices, self.A.factor * tensor.factor, 1.0)

        C = Tensor(pyambit.TensorType.CoreTensor, "C", [])
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
        positions = [i for i, x in enumerate(self.indices) if x == index]
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

                tAB = Tensor.build(A.tensor.dtype, A.tensor.name + " * " + B.tensor.name, dims)

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

                tAB = Tensor.build(A.tensor.dtype, A.tensor.name + " * " + B.tensor.name, dims)

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
           return self.tensor.__array_interface__()
       else:
           raise TypeError('Only CoreTensor tensors can be converted to ndarrays.')

    def __getitem__(self, indices):

        if isinstance(indices, str):
            return LabeledTensor(self.tensor, indices)
        else:
            return SlicedTensor(self, _parse_slices(indices, self.dims))

    def __setitem__(self, indices_str, value):

        if isinstance(value, SlicedTensor):
            if self.tensor == value.tensor:
                raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise RuntimeError("SlicedTensor::__setitem__: Sliced tensors do not have same rank")
            if isinstance(indices_str, str):
                raise TypeError("SlicedTensor::__setitem__: Slice cannot be a string")

            indices = _parse_slices(indices_str, self.dims)
            self.tensor.slice(value.tensor.tensor, indices, value.range, value.factor, 0.0)

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

                tAB = Tensor.build(A.tensor.dtype, A.tensor.name + " * " + B.tensor.name, dims)

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
        """
        Returns the raw data vector underlying the tensor object if the
        underlying tensor object supports a raw data vector. This is only the
        case if the underlying tensor is of type CoreTensor.

        This routine is intended to facilitate rapid filling of data into a
        CoreTensor buffer tensor, following which the user may stripe the buffer
        tensor into a DiskTensor or DistributedTensor tensor via slice operations.

        If a vector is successfully returned, it points to the unrolled data o
        the tensor, with the right-most dimensions running fastest and left-mo
        dimensions running slowest.

        Example successful use case:
         Tensor A = Tensor::build(CoreTensor, "A3", {4,5,6});
         std::vector<double>& Av = A.data();
         double* Ap = Av.data(); // In case the raw pointer is needed
         In this case, Av[0] = A(0,0,0), Av[1] = A(0,0,1), etc.

         Tensor B = Tensor::build(DiskTensor, "B3", {4,5,6});
         std::vector<double>& Bv = B.data(); // throws

        :return: data pointer, if tensor object supports it
        """
        return self.tensor.data()

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

    def _check_other(self, value):
        if self.tensor == value.tensor:
            raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
        if self.tensor.rank != value.tensor.rank:
            raise RuntimeError("SlicedTensor::__setitem__: Sliced tensors do not have same rank")

    def __iadd__(self, value):
        if isinstance(value, SlicedTensor):
            self._check_other(value)
            self.tensor.slice(value.tensor, self.range, value.range, value.factor, 1.0)

            return None
        else:
            raise NotImplementedError("SlicedTensor.__iadd__(%s) is not implemented" % (type(other)))


    def __isub__(self, value):
        if isinstance(value, SlicedTensor):
            self._check_other(value)
            self.tensor.slice(value.tensor, self.range, value.range, -value.factor, 1.0)

            return None
        else:
            raise NotImplementedError("SlicedTensor.__isub__(%s) is not implemented" % (type(other)))


    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return SlicedTensor(self.tensor, self.range, other * self.factor)
        else:
            raise NotImplementedError("SlicedTensor.__mul__(%s) is not implemented" % (type(other)))

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return SlicedTensor(self.tensor, self.range, other * self.factor)
        else:
            raise NotImplementedError("SlicedTensor.__rmul__(%s) is not implemented" % (type(other)))

    def __neg__(self):
        return SlicedTensor(self.tensor, self.range, -self.factor)

