from . import pyambit
from .pyambit import Tensor, EigenvalueOrder
import numbers

class LabeledTensorProduct:
    def __init__(self, left, right):
        self.factor = 1.0
        self.tensors = []
        self.tensors.append(left)
        self.tensors.append(right)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            self.tensors[0] *= other
            return self
        else:
            return NotImplemented

class LabeledTensorAddition:
    def __init__(self, left, right):
        self.tensors = []
        self.tensors.append(left)
        self.tensors.append(right)

class LabeledTensorDistributive:
    pass

class LabeledTensor:
    def __init__(self, t, indices):
        self.factor = 1.0
        self.tensor = t
        self.indices = pyambit.Indices.split(indices)
        self.labeledTensor = pyambit.LabeledTensor(self.tensor, self.indices, self.factor)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other
            return self
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
            self.tensor = pyambit.Tensor.build(type, name, dims)

    def __getitem__(self, indices):
        return LabeledTensor(self.tensor, indices)

    def __setitem__(self, indices_str, value):
        indices = pyambit.Indices.split(str(indices_str))

        if isinstance(value, LabeledTensorProduct):
            # This is "simple assignment"
            # Make sure C = C * A isn't being called.
            for tensor in value.tensors:
                if self.tensor is tensor.tensor:
                    raise ArithmeticError("Target cannot be present in contraction terms.")

            if len(value.tensors) != 2:
                raise ArithmeticError("Only pair-wise contractions are currently supported")

            A = value.tensors[0]
            B = value.tensors[1]

            #print("C[%s]" % (indices_str))
            self.tensor.contract(A.tensor, B.tensor, indices, A.indices, B.indices, value.factor * A.factor * B.factor, self.factor)

        elif isinstance(value, LabeledTensorAddition):

            # This is to handle C += A which gets translated into C = C + A
            if self.tensor is not value.tensors[0].tensor:
                self.tensor.permute(value.tensors[0].tensor, indices, value.tensors[0].indices, value.tensors[0].factor, 0.0)

            for tensor in value.tensors[1:]:
                if isinstance(tensor, LabeledTensor):
                    self.tensor.permute(tensor.tensor, indices, tensor.indices, tensor.factor, 1.0)
                else:
                    # recursively call set item
                    self.factor = 1.0
                    self.__setitem__(indices_str, tensor)

        elif isinstance(value, LabeledTensor):

            if self.tensor.rank != value.tensor.rank:
                raise ArithmeticError("Permuted tensors do not have same rank")

            # perform permutation
            self.tensor.permute(value.tensor, indices, value.indices, value.factor, 0.0)

        else:
            raise TypeError('Do not know how to set this type {}'.format(type(value).__name__))

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

    def gemm(self, A, B, transA, transB, nrow, ncol, nzip, ldaA, ldaB, ldaC, offA=0, offB=0, offC=0, alpha=1.0, beta=0.0):
        self.tensor.gemm(A.tensor, B.tensor, transA, transB, nrow, ncol, nzip, ldaA, ldaB, ldaC, offA, offB, offC, alpha, beta)

    def syev(self, order):
        aResults = self.tensor.syev(order)

        results = {}
        for k, v in aResults.iteritems():
            results[k] = Tensor(existing=v)

        return results

    def power(self, p, condition = 1.0e-12):
        aResult = self.tensor.power(p, condition)
        return Tensor(existing=aResult)
