import unittest
import random
import numpy as np
import ambit

a = ambit.Tensor(ambit.TensorType.kCore, "A", [5, 5])
b = ambit.Tensor(ambit.TensorType.kCore, "B", [5, 5])

ambit.initialize_random(a.tensor)
ambit.initialize_random(b.tensor)

a.printf()
b.printf()

print("a[i,j] = b[i,j]")
a["i,j"] = b["i,j"]
a.printf()

print("a[i,j] = 2 * b[i,j]")
a["i,j"] = 2 * b["i,j"]
a.printf()


print("a[i,j] += b[i,j]")
a["i,j"] += b["i,j"]
a.printf()

print("a[i,j] -= b[i,j]")
a["i,j"] -= b["i,j"]
a.printf()

print("a[i,j] = b[i,j] + b[i,j]")
a["i,j"] = b["i,j"] + b["i,j"]
a.printf()

print("a[i,j] = b[i,j] - b[i,j]")
a["i,j"] = b["i,j"] - b["i,j"]
a.printf()

c = ambit.Tensor(ambit.TensorType.kCore, "C", [5, 5])
ambit.initialize_random(c.tensor)

print("a[i,j] = b[i,k] * c[k,j]")
a["i,j"] = b["i,k"] * c["k,j"]
a.printf()

print("a[i,j] = b[i,k] * c[k,j] * 2")
a["i,j"] = b["i,k"] * c["k,j"] * 2
a.printf()

print("a[i,j] = 2 * b[i,k] * c[k,j]")
a["i,j"] = 2 * b["i,k"] * c["k,j"]
a.printf()

data = a.tensor.data()
for val in data:
    print("value %f" % val)
    val += 1
data[0] += 1
a.printf()

class TestOperatorOverloading(unittest.TestCase):

    def build_and_fill(self, name, dims):
        T = ambit.Tensor(ambit.TensorType.kCore, name, dims)
        N = np.empty(T.tensor.numel)

        data = T.tensor.data()
        for i, s in enumerate(data):
            value = random.random()
            data[i] = value
            N[i] = value

        return [T, N.reshape(dims)]

    def compare(self, aC, nC):
        dataA = aC.tensor.data()
        dataN = nC.reshape(aC.tensor.numel)

        for i, s in enumerate(dataA):
            if abs(dataA[i] - dataN[i]) > 1.0e-15:
                self.assertEqual(dataA[i], dataN[i], "values don't match")


    def setUp(self):
        random.seed()

        self.ni = 9
        self.nj = 6
        self.nk = 7

    def test_Cij_equal_Aik_Bkj(self):
        [aA, nA] = self.build_and_fill("A", [self.ni, self.nk])
        [aB, nB] = self.build_and_fill("B", [self.nk, self.nj])
        [aC, nC] = self.build_and_fill("C", [self.ni, self.nj])

        aC["ij"] = aA["ik"] * aB["kj"]

        nC = np.einsum('ik,kj', nA, nB)

        self.compare(aC, nC)

    def test_Cij_plus_equal_Aik_Bkj(self):
        [aA, nA] = self.build_and_fill("A", [self.ni, self.nk])
        [aB, nB] = self.build_and_fill("B", [self.nk, self.nj])
        [aC, nC] = self.build_and_fill("C", [self.ni, self.nj])

        aC["ij"] += aA["ik"] * aB["kj"]

        nC += np.einsum('ik,kj', nA, nB)

        self.compare(aC, nC)

    def test_Cij_minus_equal_Aik_Bkj(self):
        [aA, nA] = self.build_and_fill("A", [self.ni, self.nk])
        [aB, nB] = self.build_and_fill("B", [self.nk, self.nj])
        [aC, nC] = self.build_and_fill("C", [self.ni, self.nj])

        aC["ij"] -= aA["ik"] * aB["kj"]

        nC -= np.einsum('ik,kj', nA, nB)

        self.compare(aC, nC)

    def test_C_equal_A(self):
        [aA, nA] = self.build_and_fill("A", [self.ni, self.nj])
        [aB, nB] = self.build_and_fill("B", [self.ni, self.nj])

        aA["i,j"] = aB["i,j"]

        nA = nB

        self.compare(aA, nA)

if __name__ == '__main__':
    unittest.main()
