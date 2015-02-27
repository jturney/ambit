from . import pyambit
from .pyambit import Tensor
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
    def __init__(self, tensorType, name, dims):
        self.factor = 0.0
        self.tensor = pyambit.Tensor.build(tensorType, name, dims)

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
