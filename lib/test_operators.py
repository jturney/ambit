import unittest
import random
import ambit

# a = ambit.Tensor(ambit.TensorType.kCore, "A", [5, 5])
# b = ambit.Tensor(ambit.TensorType.kCore, "B", [5, 5])
#
# ambit.initialize_random(a.tensor)
# ambit.initialize_random(b.tensor)
#
# a.printf()
# b.printf()
#
# print("a[i,j] = b[i,j]")
# a["i,j"] = b["i,j"]
# a.printf()
#
# print("a[i,j] = 2 * b[i,j]")
# a["i,j"] = 2 * b["i,j"]
# a.printf()
#
#
# print("a[i,j] += b[i,j]")
# a["i,j"] += b["i,j"]
# a.printf()
#
# print("a[i,j] -= b[i,j]")
# a["i,j"] -= b["i,j"]
# a.printf()
#
# print("a[i,j] = b[i,j] + b[i,j]")
# a["i,j"] = b["i,j"] + b["i,j"]
# a.printf()
#
# print("a[i,j] = b[i,j] - b[i,j]")
# a["i,j"] = b["i,j"] - b["i,j"]
# a.printf()
#
# c = ambit.Tensor(ambit.TensorType.kCore, "C", [5, 5])
# ambit.initialize_random(c.tensor)
#
# print("a[i,j] = b[i,k] * c[k,j]")
# a["i,j"] = b["i,k"] * c["k,j"]
# a.printf()
#
# print("a[i,j] = b[i,k] * c[k,j] * 2")
# a["i,j"] = b["i,k"] * c["k,j"] * 2
# a.printf()
#
# print("a[i,j] = 2 * b[i,k] * c[k,j]")
# a["i,j"] = 2 * b["i,k"] * c["k,j"]
# a.printf()
#
# data = a.tensor.data()
# for val in data:
#     print("value %f" % val)
#     val += 1
# data[0] += 1
# a.printf()

class TestOperatorOverloading(unittest.TestCase):

    def build_and_fill2(self, name, dims):
        T = ambit.Tensor(ambit.TensorType.kCore, name, dims)
        N = [[0 for x in range(dims[1])] for x in range(dims[0])]

        data = T.tensor.data()
        for r in range(dims[0]):
            for c in range(dims[1]):
                value = random.random()

                data[r * dims[1] + c] = value
                N[r][c] = value

        return [T, N]

    def difference2(self, T, N):
        max_diff = 0.0
        data = T.tensor.data()
        dims = T.dims
        for r in range(dims[0]):
            for c in range(dims[1]):
                Tvalue = data[r*dims[1] + c]
                Nvalue = N[r][c]

                diff = abs(Tvalue - Nvalue)
                max_diff = max(max_diff, diff)

        return max_diff

    def build_and_fill3(self, name, dims):
        T = ambit.Tensor(ambit.TensorType.kCore, name, dims)
        N = [[[0 for x in range(dims[2])] for x in range(dims[1])] for x in range(dims[0])]

        data = T.tensor.data()
        for p in range(dims[0]):
            for q in range(dims[1]):
                for r in range(dims[2]):
                    value = random.random()

                    data[p*dims[1]*dims[2] + q*dims[2] + r] = value
                    N[p][q][r] = value

        return [T, N]

    def difference3(self, T, N):
        max_diff = 0.0
        data = T.tensor.data()
        dims = T.dims
        for p in range(dims[0]):
            for q in range(dims[1]):
                for r in range(dims[2]):
                    Tvalue = data[p*dims[1]*dims[2] + q*dims[2] + r]
                    Nvalue = N[p][q][r]

                    diff = abs(Tvalue - Nvalue)
                    max_diff = max(max_diff, diff)

        return max_diff

    def build_and_fill4(self, name, dims):
        T = ambit.Tensor(ambit.TensorType.kCore, name, dims)
        N = [[[[0 for x in range(dims[3])] for x in range(dims[2])] for x in range(dims[1])] for x in range(dims[0])]

        data = T.tensor.data()
        for p in range(dims[0]):
            for q in range(dims[1]):
                for r in range(dims[2]):
                    for s in range(dims[3]):
                        value = random.random()

                        data[p*dims[1]*dims[2]*dims[3] + q*dims[2]*dims[3] + r*dims[3] + s] = value
                        N[p][q][r][s] = value

        return [T, N]

    def difference4(self, T, N):
        max_diff = 0.0
        data = T.tensor.data()
        dims = T.dims
        for p in range(dims[0]):
            for q in range(dims[1]):
                for r in range(dims[2]):
                    for s in range(dims[3]):
                        Tvalue = data[p*dims[1]*dims[2]*dims[3] + q*dims[2]*dims[3] + r*dims[3] + s]
                        Nvalue = N[p][q][r][s]

                        diff = abs(Tvalue - Nvalue)
                        max_diff = max(max_diff, diff)

        return max_diff

    def build_and_fill(self, name, dims):
        if len(dims) == 2:
            return self.build_and_fill2(name, dims)
        elif len(dims) == 3:
            return self.build_and_fill3(name, dims)
        elif len(dims) == 4:
            return self.build_and_fill4(name, dims)
        else:
            raise RuntimeError("Don't know how to build tensors of order %d" % len(dims))

    def difference(self, T, N):
        dims = T.dims
        if len(dims) == 2:
            return self.difference2(T, N)
        elif len(dims) == 3:
            return self.difference3(T, N)
        elif len(dims) == 4:
            return self.difference4(T, N)
        else:
            raise RuntimeError("Don't know how to difference tensors of order %d" % len(dims))

    def test_Cij_equal_2Aij(self):
        ni = 9
        nj = 6
        [aA, nA] = self.build_and_fill2("A", [ni, nj])
        [aC, nC] = self.build_and_fill2("C", [ni, nj])

        aC["i,j"] = 2.0*aA["i,j"]

        for i in range(ni):
            for j in range(nj):
                nC[i][j] = 2.0 * nA[i][j]

        self.assertAlmostEqual(0.0, self.difference2(aC, nC), places=12)

    def test_Cij_plus_equal_2Aij(self):
        ni = 9
        nj = 6
        [aA, nA] = self.build_and_fill2("A", [ni, nj])
        [aC, nC] = self.build_and_fill2("C", [ni, nj])

        aC["i,j"] += 2.0*aA["i,j"]

        for i in range(ni):
            for j in range(nj):
                nC[i][j] += 2.0 * nA[i][j]

        self.assertAlmostEqual(0.0, self.difference2(aC, nC), places=12)

    def test_Cij_minus_equal_2Aij(self):
        ni = 9
        nj = 6
        [aA, nA] = self.build_and_fill2("A", [ni, nj])
        [aC, nC] = self.build_and_fill2("C", [ni, nj])

        aC["i,j"] -= 2.0*aA["i,j"]

        for i in range(ni):
            for j in range(nj):
                nC[i][j] -= 2.0 * nA[i][j]

        self.assertAlmostEqual(0.0, self.difference2(aC, nC), places=12)

    def test_Cij_times_equal_2(self):
        ni = 9
        nj = 6
        [aC, nC] = self.build_and_fill2("C", [ni, nj])

        aC["i,j"] *= 2.0

        for i in range(ni):
            for j in range(nj):
                nC[i][j] *= 2.0

        self.assertAlmostEqual(0.0, self.difference2(aC, nC), places=12)

    def test_Cij_divide_equal_2(self):
        ni = 9
        nj = 6
        [aC, nC] = self.build_and_fill2("C", [ni, nj])

        aC["i,j"] /= 2.0

        for i in range(ni):
            for j in range(nj):
                nC[i][j] /= 2.0

        self.assertAlmostEqual(0.0, self.difference2(aC, nC), places=12)

    def test_Cij_equal_Aij(self):
        ni = 9
        nj = 6
        [aA, nA] = self.build_and_fill2("A", [ni, nj])
        [aC, nC] = self.build_and_fill2("C", [ni, nj])

        aC["i,j"] = aA["i,j"]

        for i in range(ni):
            for j in range(nj):
                nC[i][j] = nA[i][j]

        self.assertAlmostEqual(0.0, self.difference2(aC, nC), places=12)

    def test_Cijkl_equal_Akijl(self):
        ni = 9
        nj = 6
        nk = 5
        nl = 4

        [aA, nA] = self.build_and_fill("A", [nk, ni, nj, nl])
        [aC, nC] = self.build_and_fill("C", [ni, nj, nk, nl])

        aC["ijkl"] = aA["kijl"]

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        nC[i][j][k][l] = nA[k][i][j][l]

        self.assertAlmostEqual(0.0, self.difference(aC, nC), places=12)

    def test_Cijkl_equal_Akilj(self):
        ni = 9
        nj = 6
        nk = 5
        nl = 4

        [aA, nA] = self.build_and_fill("A", [nk, ni, nl, nj])
        [aC, nC] = self.build_and_fill("C", [ni, nj, nk, nl])

        aC["ijkl"] = aA["kilj"]

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        nC[i][j][k][l] = nA[k][i][l][j]

        self.assertAlmostEqual(0.0, self.difference(aC, nC), places=12)

    @unittest.expectedFailure
    def test_Cij_equal_Cij(self):
        ni = 9
        nj = 6

        [aC, nC] = self.build_and_fill("C", [ni, nj])

        aC["ij"]  = aC["ij"]

    def test_Cij_plus_equal_Aik_Bkj(self):
        ni = 9
        nj = 6
        nk = 7

        [aA, nA] = self.build_and_fill("A", [ni, nk])
        [aB, nB] = self.build_and_fill("B", [nk, nj])
        [aC, nC] = self.build_and_fill("C", [ni, nj])

        aC["ij"] += aA["ik"] * aB["kj"]

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    nC[i][j] += nA[i][k] * nB[k][j]

        self.assertAlmostEqual(0.0, self.difference(aC, nC), places=12)

    def test_Cij_minus_equal_Aik_Bkj(self):
        ni = 9
        nj = 6
        nk = 7

        [aA, nA] = self.build_and_fill("A", [ni, nk])
        [aB, nB] = self.build_and_fill("B", [nk, nj])
        [aC, nC] = self.build_and_fill("C", [ni, nj])

        aC["ij"] -= aA["ik"] * aB["kj"]

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    nC[i][j] -= nA[i][k] * nB[k][j]

        self.assertAlmostEqual(0.0, self.difference(aC, nC), places=12)
if __name__ == '__main__':
    unittest.main()
