import unittest
import random
import ambit

class TestOperatorOverloading(unittest.TestCase):

    def setUp(self):
        ambit.initialize()

    def tearDown(self):
        ambit.finalize()

    def build_and_fill2(self, name, dims):
        T = ambit.Tensor(ambit.TensorType.CoreTensor, name, dims)
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
        T = ambit.Tensor(ambit.TensorType.CoreTensor, name, dims)
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
        T = ambit.Tensor(ambit.TensorType.CoreTensor, name, dims)
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

    def test_Cij_equal_Aik_Bkj(self):
        ni = 9
        nj = 6
        nk = 7

        [aA, nA] = self.build_and_fill("A", [ni, nk])
        [aB, nB] = self.build_and_fill("B", [nk, nj])
        [aC, nC] = self.build_and_fill("C", [ni, nj])

        aC["ij"] = aA["ik"] * aB["kj"]

        for i in range(ni):
            for j in range(nj):
                nC[i][j] = 0.0
                for k in range(nk):
                    nC[i][j] += nA[i][k] * nB[k][j]

        self.assertAlmostEqual(0.0, self.difference(aC, nC), places=12)

    def test_Cij_equal_Aik_Bjk(self):
        ni = 9
        nj = 6
        nk = 7

        [aA, nA] = self.build_and_fill("A", [ni, nk])
        [aB, nB] = self.build_and_fill("B", [nj, nk])
        [aC, nC] = self.build_and_fill("C", [ni, nj])

        aC["ij"] = aA["ik"] * aB["jk"]

        for i in range(ni):
            for j in range(nj):
                nC[i][j] = 0.0
                for k in range(nk):
                    nC[i][j] += nA[i][k] * nB[j][k]

        self.assertAlmostEqual(0.0, self.difference(aC, nC), places=12)

    def test_Cijkl_plus_equal_Aijab_Bklab(self):
        ni = 9
        nj = 6
        nk = 7
        nl = 9
        na = 6
        nb = 7

        [A, nA] = self.build_and_fill("A", [ni, nj, na, nb])
        [B, nB] = self.build_and_fill("B", [nk, nl, na, nb])
        [C, nC] = self.build_and_fill("C", [ni, nj, nk, nl])

        C["ijkl"] += A["ijab"] * B["klab"]

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        for a in range(na):
                            for b in range(nb):
                                nC[i][j][k][l] += nA[i][j][a][b] * nB[k][l][a][b]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_Cij_plus_equal_Aiabc_B_jabc(self):
        ni = 9
        nj = 6
        na = 6
        nb = 7
        nc = 8

        [A, nA] = self.build_and_fill("A", [ni, na, nb, nc])
        [B, nB] = self.build_and_fill("B", [nj, na, nb, nc])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        C["ij"] += A["iabc"] * B["jabc"]

        for i in range(ni):
            for j in range(nj):
                for a in range(na):
                    for b in range(nb):
                        for c in range(nc):
                            nC[i][j] += nA[i][a][b][c] * nB[j][a][b][c]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    @unittest.expectedFailure
    def test_E_abcd_plus_equal_Aijab_Bklcd_C_jl_D_ik(self):
        ni = 9
        nj = 6
        nk = 7
        nl = 9
        na = 6
        nb = 7
        nc = 6
        nd = 7

        [A, nA] = self.build_and_fill("A", [ni, nj, na, nb])
        [B, nB] = self.build_and_fill("B", [nk, nl, nc, nd])
        [C, nC] = self.build_and_fill("C", [nj, nl])
        [D, nD] = self.build_and_fill("D", [ni, nk])
        [E, nE] = self.build_and_fill("E", [na, nb, nc, nd])

        E["abcd"] += A["ijab"] * B["klcd"] * C["jl"] * D["ik"]

        for a in range(na):
            for b in range(nb):
                for c in range(nc):
                    for d in range(nd):
                        for i in range(ni):
                            for j in range(ni):
                                for k in range(nk):
                                    for l in range(nl):
                                        nE[a][b][c][d] += nA[i][j][a][b] * nB[k][l][c][d] * nC[j][l] * nD[i][k]

        self.assertAlmostEqual(0.0, self.difference(E, nE), places=12)

    def test_Cilkj_equal_Aibaj_Bblak(self):
        ni = 9
        nj = 6
        nk = 7
        nl = 9
        na = 6
        nb = 7

        [A, nA] = self.build_and_fill("A", [ni, nb, na, nj])
        [B, nB] = self.build_and_fill("B", [nb, nl, na, nk])
        [C, nC] = self.build_and_fill("C", [ni, nl, nk, nj])

        C["ilkj"] = A["ibaj"] * B["blak"]

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        nC[i][l][k][j] = 0.0
                        for a in range(na):
                            for b in range(nb):
                                nC[i][l][k][j] += nA[i][b][a][j] * nB[b][l][a][k]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_Cljik_equal_Abija_Blbak(self):
        ni = 9
        nj = 6
        nk = 7
        nl = 9
        na = 6
        nb = 7

        [A, nA] = self.build_and_fill("A", [nb, ni, nj, na])
        [B, nB] = self.build_and_fill("B", [nl, nb, na, nk])
        [C, nC] = self.build_and_fill("C", [nl, nj, ni, nk])

        C["ljik"] += A["bija"] * B["lbak"]

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for l in range(nl):
                        for a in range(na):
                            for b in range(nb):
                                nC[l][j][i][k] += nA[b][i][j][a] * nB[l][b][a][k]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_Cij_equal_Aij_plus_Bij(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        C["ij"] = A["ij"] + B["ij"]

        for i in range(ni):
            for j in range(nj):
                nC[i][j] = nA[i][j] + nB[i][j]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_Dij_equal_Aij_plus_Bij_plus_Cij(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])
        [D, nD] = self.build_and_fill("D", [ni, nj])

        D["ij"] = A["ij"] +  B["ij"] + C["ij"]

        for i in range(ni):
            for j in range(nj):
                nD[i][j] = nA[i][j] + nB[i][j] + nC[i][j]

        self.assertAlmostEqual(0.0, self.difference(D, nD), places=12)

    def test_Cij_equal_Aij_minus_5Bij(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        C["ij"] = A["ij"] - 5.0*B["ij"]

        for i in range(ni):
            for j in range(nj):
                nC[i][j] = nA[i][j] - 5.0* nB[i][j]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_Dij_equal_Aij_minus_Bij_plus_2Cij(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])
        [D, nD] = self.build_and_fill("D", [ni, nj])

        D["ij"] = A["ij"] -  B["ij"] + 2.0*C["ij"]

        for i in range(ni):
            for j in range(nj):
                nD[i][j] = nA[i][j] - nB[i][j] + 2.0*nC[i][j]

        self.assertAlmostEqual(0.0, self.difference(D, nD), places=12)

    def test_Dij_equal_Aij_times_2Bij_plus_2Cij(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])
        [D, nD] = self.build_and_fill("D", [ni, nj])

        D["ij"] = A["ij"] * (2.0*B["ij"] - C["ij"])

        for i in range(ni):
            for j in range(nj):
                nD[i][j] = nA[i][j] * (2.0* nB[i][j] - nC[i][j])

        self.assertAlmostEqual(0.0, self.difference(D, nD), places=12)

    def test_Dij_equal_Bij_plus_Cij_times_Aij(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])
        [D, nD] = self.build_and_fill("D", [ni, nj])

        D["ij"] = (2.0*B["ij"] - C["ij"]) * A["ij"]

        for i in range(ni):
            for j in range(nj):
                nD[i][j] = nA[i][j] * (2.0* nB[i][j] - nC[i][j])

        self.assertAlmostEqual(0.0, self.difference(D, nD), places=12)

    def test_F_equal_D_times_2g_minus_g(self):
        ni = 9
        nj = 9
        nk = 9
        nl = 9

        [F, nF] = self.build_and_fill("F", [ni, nj])
        [D, nD] = self.build_and_fill("D", [nk, nl])
        [g, ng] = self.build_and_fill("g", [ni, nj, nk, nl])

        F["ij"] = D["kl"] * (2.0 * g["ijkl"] - g["ikjl"])

        for i in range(ni):
            for j in range(nj):
                nF[i][j] = 0.0
                for k in range(nk):
                    for l in range(nl):
                        nF[i][j] += nD[k][l] * (2.0 * ng[i][j][k][l] - ng[i][k][j][l])

        self.assertAlmostEqual(0.0, self.difference(F, nF), places=12)

    def test_Dij_equal_2_times_Aij_plus_Bij(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        C["ij"] = 2.0 * (A["ij"] - B["ij"])

        for i in range(ni):
            for j in range(nj):
                nC[i][j] = 2.0 * (nA[i][j] - nB[i][j])

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_Dij_equal_negate_Aij_plus_Bij(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        C["ij"] = - (A["ij"] - B["ij"])

        for i in range(ni):
            for j in range(nj):
                nC[i][j] = - (nA[i][j] - nB[i][j])

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_dot_product1(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])

        C = float(A["ij"] * B["ij"])
        nC = 0.0

        for i in range(ni):
            for j in range(nj):
                nC += nA[i][j] * nB[i][j]

        self.assertAlmostEqual(0.0, C - nC, places=12)

    @unittest.expectedFailure
    def test_dot_product2(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])


        C = float(A["ij"] * B["ik"])
        nC = 0.0

        for i in range(ni):
            for j in range(nj):
                nC += nA[i][j] * nB[i][j]

        self.assertAlmostEqual(0.0, C - nC, places=12)

    @unittest.expectedFailure
    def test_dot_product3(self):
        ni = 9
        nj = 6
        nk = 5

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nk])

        C = float(A["ij"] * B["ij"])
        nC = 0.0

        for i in range(ni):
            for j in range(nj):
                nC += nA[i][j] * nB[i][j]

        self.assertAlmostEqual(0.0, C - nC, places=12)

    def test_dot_product4(self):
        ni = 9
        nj = 6

        [A, nA] = self.build_and_fill("A", [ni, nj])
        [B, nB] = self.build_and_fill("B", [ni, nj])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        D = float(A["ij"] * (B["ij"] + C["ij"]))
        nD = 0.0

        for i in range(ni):
            for j in range(nj):
                nD += nA[i][j] * (nB[i][j] + nC[i][j])

        self.assertAlmostEqual(0.0, D - nD, places=12)

    def test_slice1(self):
        ni = 7
        nj = 7
        nk = 7
        nl = 7

        [A, nA] = self.build_and_fill("A", [nk, nl])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        Ainds = [[0, 4], [2, 6]]
        Cinds = [[1, 5], [0, 4]]

        C[Cinds] = A[Ainds]

        for i in range(Cinds[0][1] - Cinds[0][0]):
            for j in range(Cinds[1][1] - Cinds[1][0]):
                nC[i + Cinds[0][0]][j + Cinds[1][0]] = nA[i + Ainds[0][0]][j + Ainds[1][0]]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_slice2(self):
        ni = 7
        nj = 7
        nk = 7
        nl = 7

        [A, nA] = self.build_and_fill("A", [nk, nl])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        Ainds = [[0, 4], [2, 6]]
        Cinds = [[1, 5], [0, 4]]

        C[Cinds] += A[Ainds]

        for i in range(Cinds[0][1] - Cinds[0][0]):
            for j in range(Cinds[1][1] - Cinds[1][0]):
                nC[i + Cinds[0][0]][j + Cinds[1][0]] += nA[i + Ainds[0][0]][j + Ainds[1][0]]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_slice3(self):
        ni = 7
        nj = 7
        nk = 7
        nl = 7

        [A, nA] = self.build_and_fill("A", [nk, nl])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        Ainds = [[0, 4], [2, 6]]
        Cinds = [[1, 5], [0, 4]]

        C[Cinds] -= A[Ainds]

        for i in range(Cinds[0][1] - Cinds[0][0]):
            for j in range(Cinds[1][1] - Cinds[1][0]):
                nC[i + Cinds[0][0]][j + Cinds[1][0]] -= nA[i + Ainds[0][0]][j + Ainds[1][0]]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_slice4(self):
        ni = 7
        nj = 7
        nk = 7
        nl = 7

        [A, nA] = self.build_and_fill("A", [nk, nl])
        [C, nC] = self.build_and_fill("C", [ni, nj])

        Ainds = [[0, 4], [2, 6]]
        Cinds = [[1, 5], [0, 4]]

        C[1:5, 0:4] -= A[0:4, 2:6]

        for i in range(Cinds[0][1] - Cinds[0][0]):
            for j in range(Cinds[1][1] - Cinds[1][0]):
                nC[i + Cinds[0][0]][j + Cinds[1][0]] -= nA[i + Ainds[0][0]][j + Ainds[1][0]]

        self.assertAlmostEqual(0.0, self.difference(C, nC), places=12)

    def test_slice_bounds(self):
        [C, nC] = self.build_and_fill("C", [10, 10])
        with self.assertRaises(RuntimeError):
            C[1:5, 0:4, :3]
        with self.assertRaises(RuntimeError):
            C[:, :, :]
        with self.assertRaises(RuntimeError):
            C[:, :, :, :]
        with self.assertRaises(RuntimeError):
            C[slice(5), slice(5), slice(5)]

    def test_slice_step(self):
        [C, nC] = self.build_and_fill("C", [10, 10])
        with self.assertRaises(ValueError):
            C[1:5:2, 1:5]
        with self.assertRaises(ValueError):
            C[1:5, 1:5:2]
        with self.assertRaises(ValueError):
            C[1:5:2]
        with self.assertRaises(ValueError):
            C[:, 1:5:2]

if __name__ == '__main__':
    unittest.main()
