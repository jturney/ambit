import ambit
from numbers import Real

class MultTensor:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class AddTensor:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class SubtractTensor(AddTensor):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.right.factor = -self.right.factor

class IndexedTensor:
    def __init__(self, t, indices):
        self.factor = 1.0
        self.tensor = t
        self.indices = indices

    def __mul__(self, other):
        return MultTensor(self, other)

    def __add__(self, other):
        return AddTensor(self, other)

    def __sub__(self, other):
        return SubtractTensor(self, other)

    def __rmul__(self, other):
        if isinstance(other, Real):
            self.factor *= other
        else:
            return NotImplemented

class Tensor:
    def __init__(self, world, dims, syms):
        self.factor = 1.0
        self.world = world
        self.dims = dims
        self.syms = syms
        self.tensor = ambit.Tensor(world, dims, syms)

    def __getitem__(self, indices):
        return IndexedTensor(self.tensor, indices)

    def __setitem__(self, indices, value):
        indices = str(indices)
        if isinstance(value, MultTensor):
            self.tensor.contract(value.left.factor, value.left.tensor, value.left.indices,
                                 value.right.tensor, value.right.indices,
                                 0.0, indices)
        elif isinstance(value, AddTensor):
            self.tensor.sum(value.left.factor, value.left.tensor, value.left.indices, 0.0, indices)
            self.tensor.sum(value.right.factor, value.right.tensor, value.right.indices, 1.0, indices)
        elif isinstance(value, IndexedTensor):
            self.tensor.sort(value.factor, value.tensor, value.indices, indices)
        else:
            print("I don't know what to do!")
            raise TypeError('Do not know how to set this type {}'.format(type(value).__name__))

