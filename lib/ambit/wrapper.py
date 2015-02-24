from . import ambit
from .ambit import Tensor
from numbers import Real

class LabeledTensorProduct:
    def __init__(self, left, right):
        self.left = left
        self.right = right

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
        self.indices = ambit.Indices.split(indices)
        self.labeledTensor = ambit.LabeledTensor(self.tensor, self.indices, self.factor)

    def T(self):
        return self.tensor

    def __mul__(self, other):
        return MultTensor(self, other)

    def __add__(self, other):
        return LabeledTensorAddition(self, other)

    def __sub__(self, other):
        other.factor *= -1.0
        return LabeledTensorAddition(self, other)

    def __rmul__(self, other):
        if isinstance(other, Real):
            self.factor *= other
        else:
            return NotImplemented

class Tensor:
    def __init__(self, tensorType, name, dims):
        self.tensor = ambit.Tensor.build(tensorType, name, dims)

    def __getitem__(self, indices):
        return LabeledTensor(self.tensor, indices)

    def __setitem__(self, indices, value):
        indices = ambit.Indices.split(str(indices))
        if isinstance(value, LabeledTensorProduct):
            print("In Tensor::__setitem__ with LabeledTensorProduct")

            return NotImplemented

        elif isinstance(value, LabeledTensorAddition):

            if self.tensor is not value.tensors[0].tensor:
                self.tensor.permute(value.tensors[0].tensor, indices, value.tensors[0].indices, value.tensors[0].factor, 0.0)

            for tensor in value.tensors[1:]:
                self.tensor.permute(tensor.tensor, indices, tensor.indices, tensor.factor, 1.0)

        elif isinstance(value, LabeledTensor):
            self.tensor.permute(value.T(), indices, value.indices, value.factor, 0.0)

        else:
            raise TypeError('Do not know how to set this type {}'.format(type(value).__name__))

    def printf(self):
        self.tensor.printf()
