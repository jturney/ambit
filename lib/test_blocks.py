import unittest
import random
import math
import ambit

class TestBlocks(unittest.TestCase):

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

                # if abs(diff) > 1.0E-12:
                #     print("r %d c %d %f\n" % (r, c, diff))

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

    def test_mo_space(self):
        alpha_occ = ambit.MOSpace("o", "i,j,k,l",[0,1,2,3,4],ambit.SpinType.AlphaSpin)
        alpha_vir = ambit.MOSpace("v", "a,b,c,d",[5,6,7,8,9],ambit.SpinType.AlphaSpin)

    def test_add_mo_space(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l",[0,1,2,3,4],ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d",[5,6,7,8,9],ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g","p,q,r,s,t",{"o","v"})

    @unittest.expectedFailure
    def test_add_mo_space_nonexisting_space(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l",[0,1,2,3,4],ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g","p,q,r,s,t",{"o","v"})

    @unittest.expectedFailure
    def test_add_mo_space_repeated_index1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,i", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

    @unittest.expectedFailure
    def test_add_mo_space_repeated_index2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,a", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

    @unittest.expectedFailure
    def test_add_mo_space_repeated_index3(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s,c", ["o", "v"])

    @unittest.expectedFailure
    def test_add_mo_space_no_name1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)

    @unittest.expectedFailure
    def test_add_mo_space_no_name2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("", "p,q,r,s", ["o", "v"])

    @unittest.expectedFailure
    def test_add_mo_space_no_index1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "", [0,1,2,3,4], ambit.SpinType.AlphaSpin)

    @unittest.expectedFailure
    def test_add_mo_space_no_index2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "", ["o", "v"])

    @unittest.expectedFailure
    def test_add_mo_space_no_mos(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [], ambit.SpinType.AlphaSpin)

    @unittest.expectedFailure
    def test_add_mo_space_repeated_space1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("o", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

    @unittest.expectedFailure
    def test_add_mo_space_repeated_space2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("o", "p,q,r,s,c", ["o", "v"])

    def test_block_creation1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T", ["oo", "vv"])

    def test_block_creation2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])
        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "F", ["gg"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "V", ["gggg"])

    def test_block_creation3(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("c", "m,n", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("a", "u,v", [3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("h", "i,j,k,l", ["c", "a"])
        ambit.BlockedTensor.add_composite_mo_space("p", "a,b,c,d", ["a", "v"])
        ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T1", ["hp"])
        ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T2", ["hhpp"])

    @unittest.expectedFailure
    def test_block_creation_bad_rank(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T", ["oo", "ovv"])

    def test_block_norm_1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T2", ["aavv"])
        T2.set(0.5)
        self.assertAlmostEqual(112.5, T2.norm(1), places=12)

    def test_block_norm_2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T2", ["aavv"])
        T2.set(0.5)
        self.assertAlmostEqual(7.5, T2.norm(2), places=12)

    def test_block_norm_3(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T2", ["aavv"])
        T2.set(0.5)
        self.assertAlmostEqual(0.5, T2.norm(0), places=12)

    def test_block_zero(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T2", ["aavv"])
        T2.set(0.5)
        T2.zero()
        self.assertAlmostEqual(0.0, T2.norm(2), places=12)

    def test_block_scale(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T2", ["aavv"])
        T2.set(2.0)
        T2.scale(0.25)
        self.assertAlmostEqual(7.5, T2.norm(2), places=12)

    def test_block_labels1(self):
        pass

    def test_block_retrieve_block1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        T = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T", ["oo", "vv"])
        T.block("oo")

    @unittest.expectedFailure
    def test_block_retrieve_block2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", {"o", "v"})
        T = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T", {"oo", "vv"})
        T.block("og")

    @unittest.expectedFailure
    def test_block_retrieve_block3(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        T = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T", ["oo", "vv"])
        T.block("ov")

    @unittest.expectedFailure
    def test_block_retrieve_block4(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        T = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "T", ["oo", "vv"])
        T.block("")

    def test_block_iterator1(self):
        pass

    def test_Cij_equal_Aji(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "vv", "ov", "vo"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "vv", "ov", "vo"])

        Aoo = A.block("oo")
        Coo = C.block("oo")

        no = 3
        nv = 5

        [Aoo_t, a2] = self.build_and_fill2("A", [no, no])
        [Coo_t, c2] = self.build_and_fill2("C", [no, no])

        Aoo["ij"] = Aoo_t["ij"]
        Coo["ij"] = Coo_t["ij"]

        C["ij"] = A["ji"]

        for i in range(no):
            for j in range(no):
                c2[i][j] = a2[j][i]

        diff_oo = self.difference2(Coo, c2)

        self.assertAlmostEqual(0.0, diff_oo, places=12)

        Aov = A.block("ov")
        Cvo = C.block("vo")

        [Aov_t, a2] = self.build_and_fill2("A", [no, nv])
        [Cvo_t, c2] = self.build_and_fill2("C", [nv, no])

        Aov["ij"] = Aov_t["ij"]
        Cvo["ij"] = Cvo_t["ij"]

        C["ai"] = A["ia"]

        for i in range(no):
            for a in range(nv):
                c2[a][i] = a2[i][a]

        diff_vo = self.difference2(Cvo, c2)

        self.assertAlmostEqual(0.0, diff_vo, places=12)

        Avv = A.block("vv")
        Cvv = C.block("vv")

        [Avv_t, a2] = self.build_and_fill2("A", [nv, nv])
        [Cvv_t, c2] = self.build_and_fill2("C", [nv, nv])

        Avv["ij"] = Avv_t["ij"]
        Cvv["ij"] = Cvv_t["ij"]

        C["ab"] = A["ab"]

        for a in range(nv):
            for b in range(nv):
                c2[a][b] = a2[a][b]

        diff_vv = self.difference2(Cvv, c2)

        self.assertAlmostEqual(0.0, diff_vv, places=12)

    def test_Cijab_plus_equal_Aaibj(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["vovo", "ovvo", "voov"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oovv", "ovvo", "voov"])

        no = 3
        nv = 5

        Avovo = A.block("vovo")
        Coovv = C.block("oovv")

        [Avovo_t, a4] = self.build_and_fill("A", [nv, no, nv, no])
        [Coovv_t, c4] = self.build_and_fill("C", [no, no, nv, nv])

        Avovo["pqrs"] = Avovo_t["pqrs"]
        Coovv["pqrs"] = Coovv_t["pqrs"]

        C["ijab"] += A["aibj"]

        for i in range(no):
            for j in range(no):
                for a in range(nv):
                    for b in range(nv):
                        c4[i][j][a][b] += a4[a][i][b][j]

        self.assertAlmostEqual(0.0, self.difference(Coovv, c4), places=12)

    def test_Cijab_plus_equal_Aaibj(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["vovo", "ovvo", "voov"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oovv", "ovvo", "voov"])

        no = 3
        nv = 5

        Aovvo = A.block("ovvo")
        Cvoov = C.block("voov")

        [Aovvo_t, a4] = self.build_and_fill("A", [no, nv, nv, no])
        [Cvoov_t, c4] = self.build_and_fill("C", [nv, no, no, nv])

        Aovvo["pqrs"] = Aovvo_t["pqrs"]
        Cvoov["pqrs"] = Cvoov_t["pqrs"]

        C["bija"] += A["jabi"]

        for i in range(no):
            for j in range(no):
                for a in range(nv):
                    for b in range(nv):
                        c4[b][i][j][a] += a4[j][a][b][i]

        self.assertAlmostEqual(0.0, self.difference(Cvoov, c4), places=12)

    def test_Cij_times_equal_double(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "vv", "ov", "vo"])

        no = 3
        nv = 5

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Aov_t, b2] = self.build_and_fill("Aov", [no, nv])
        c2 = [[0 for x in range(no)] for x in range(no)]

        A.block("oo")["pq"] = Aoo_t["pq"]
        A.block("ov")["pq"] = Aov_t["pq"]

        A["ij"] *= math.exp(1.0)

        for i in range(no):
            for j in range(no):
                c2[i][j] = math.exp(1.0) * a2[i][j]

        self.assertAlmostEqual(0.0, self.difference(A.block("oo"), c2), places=12)

        c2 = [[0 for x in range(nv)] for x in range(no)]

        A["ia"] /= math.exp(1.0)

        for i in range(no):
            for a in range(nv):
                c2[i][a] = b2[i][a] / math.exp(1.0)

        self.assertAlmostEqual(0.0, self.difference(A.block("ov"), c2), places=12)

    def test_Cip_times_equal_double(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "vv", "ov", "vo"])

        no = 3
        nv = 5

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Aov_t, b2] = self.build_and_fill("Aov", [no, nv])
        c2 = [[0 for x in range(no)] for x in range(no)]


        A.block("oo")["pq"] = Aoo_t["pq"]
        A.block("ov")["pq"] = Aov_t["pq"]

        A["ip"] *= math.exp(1.0)

        for i in range(no):
            for j in range(no):
                c2[i][j] = math.exp(1.0) * a2[i][j]

        self.assertAlmostEqual(0.0, self.difference(A.block("oo"), c2), places=12)

        c2 = [[0 for x in range(nv)] for x in range(no)]

        for i in range(no):
            for a in range(nv):
                c2[i][a] = math.exp(1.0) * b2[i][a]

        self.assertAlmostEqual(0.0, self.difference(A.block("ov"), c2), places=12)

    def test_Cij_equal_Aij_Bjk(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block("oo")["pq"] = Aoo_t["pq"]
        B.block("oo")["pq"] = Boo_t["pq"]
        C.block("oo")["pq"] = Coo_t["pq"]

        for i in range(no):
            for j in range(no):
                c2[i][j] = 0.0
                for k in range(no):
                    c2[i][j] += a2[i][k] * b2[j][k]

        C["ij"] = A["ik"] * B["jk"]

        self.assertAlmostEqual(0.0, self.difference(C.block("oo"), c2), places=12)

    def test_Cij_equal_Aip_Bjp(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k", [0,1,2,10,12], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 5
        nv = 7

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Aoo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Aoo", [no, no])
        [Aov_t, d2] = self.build_and_fill("Aoo", [no, nv])
        [Bov_t, e2] = self.build_and_fill("Aoo", [no, nv])
        [Cov_t, f2] = self.build_and_fill("Aoo", [no, nv])

        A.block("oo")["pq"] = Aoo_t["pq"]
        B.block("oo")["pq"] = Boo_t["pq"]
        C.block("oo")["pq"] = Coo_t["pq"]
        A.block("ov")["pq"] = Aov_t["pq"]
        B.block("ov")["pq"] = Bov_t["pq"]
        C.block("ov")["pq"] = Cov_t["pq"]

        for i in range(no):
            for j in range(no):
                c2[i][j] = 0.0
                for k in range(no):
                    c2[i][j] += a2[i][k] * b2[j][k]

                for a in range(nv):
                    c2[i][j] += d2[i][a] * e2[j][a]

        C["ij"] = A["ip"] * B["jp"]
        C["ab"] = A["ap"] * B["bp"]

        self.assertAlmostEqual(0.0, self.difference(C.block("oo"), c2), places=12)

    def test_Cij_equal_half_Aia_Baj(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3
        nv = 5

        [Aov_t, a2] = self.build_and_fill("Aov", [no, nv])
        [Bvo_t, b2] = self.build_and_fill("Bvo", [nv, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block("ov")["pq"] = Aov_t["pq"]
        B.block("vo")["pq"] = Bvo_t["pq"]
        C.block("oo")["pq"] = Coo_t["pq"]

        for i in range(no):
            for j in range(no):
                c2[i][j] = 0.0
                for a in range(nv):
                    c2[i][j] += 0.5 * a2[i][a] * b2[a][j]

        C["ij"] = 0.5 * A["ia"] * B["aj"]

        self.assertAlmostEqual(0.0, self.difference(C.block("oo"), c2), places=12)

    def test_Cij_plus_equal_half_Aai_Bja(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3
        nv = 5

        [Avo_t, a2] = self.build_and_fill("Avo", [nv, no])
        [Bov_t, b2] = self.build_and_fill("Bov", [no, nv])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block("vo")["pq"] = Avo_t["pq"]
        B.block("ov")["pq"] = Bov_t["pq"]
        C.block("oo")["pq"] = Coo_t["pq"]

        for i in range(no):
            for j in range(no):
                for a in range(nv):
                    c2[i][j] += 0.5 * a2[a][i] * b2[j][a]

        C["ij"] += A["ai"] * 0.5 * B["ja"]

        self.assertAlmostEqual(0.0, self.difference(C.block("oo"), c2), places=12)

    def test_Cij_minus_equal_Aik_Bjk(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3

        [Aoo_t, a2] = self.build_and_fill("Aov", [no, no])
        [Boo_t, b2] = self.build_and_fill("Bvo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block("oo")["pq"] = Aoo_t["pq"]
        B.block("oo")["pq"] = Boo_t["pq"]
        C.block("oo")["pq"] = Coo_t["pq"]

        for i in range(no):
            for j in range(no):
                for k in range(no):
                    c2[i][j] -= a2[i][k] * b2[j][k]

        C["ij"] -= A["ik"] * B["jk"]

        self.assertAlmostEqual(0.0, self.difference(C.block("oo"), c2), places=12)

    def test_chain_multiply(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])
        D = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "D", ["oo", "ov", "vo", "vv"])

        no = 3

        [Aoo_t, a2] = self.build_and_fill("Aov", [no, no])
        [Boo_t, b2] = self.build_and_fill("Bvo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])
        [Doo_t, d2] = self.build_and_fill("Doo", [no, no])

        A.block("oo")["pq"] = Aoo_t["pq"]
        B.block("oo")["pq"] = Boo_t["pq"]
        C.block("oo")["pq"] = Coo_t["pq"]
        D.block("oo")["pq"] = Doo_t["pq"]

        for i in range(no):
            for j in range(no):
                d2[i][j] = 0.0
                for k in range(no):
                    for l in range(no):
                        d2[i][j] += a2[l][j] * b2[i][k] * c2[k][l]

        D['ij'] = B['ik'] * C['kl'] * A['lj']

        self.assertAlmostEqual(0.0, self.difference(D.block("oo"), d2), places=12)

    def test_chain_multiply2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9,10,15,20], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["vvoo"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])
        D = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "D", ["oovv", "ovvo"])

        no = 5
        nv = 8

        [Avvoo_t, a4] = self.build_and_fill("Avvoo", [nv, nv, no, no])
        [Boo_t,   b2] = self.build_and_fill("Boo",   [no, no])
        [Coo_t,   c2] = self.build_and_fill("Coo",   [no, no])
        [Doovv_t, d4] = self.build_and_fill("Doovv", [no, no, nv, nv])

        A.block('vvoo')['pqrs'] = Avvoo_t['pqrs']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']
        D.block('oovv')['pqrs'] = Doovv_t['pqrs']

        for i in range(no):
            for j in range(no):
                for a in range(nv):
                    for b in range(nv):
                        d4[i][j][a][b] = 0.0
                        for k in range(no):
                            for l in range(no):
                                d4[i][j][a][b] += a4[a][b][l][j] * b2[i][k] * c2[k][l]

        D['ijab'] = B['ik'] * C['kl'] * A['ablj']

        self.assertAlmostEqual(0.0, self.difference(D.block("oovv"), d4), places=12)

    def test_Cij_equal_Aij_plus_Bij(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']

        for i in range(no):
            for j in range(no):
                c2[i][j] = a2[i][j] + b2[i][j]

        C['ij'] = A['ij'] + B['ij']

        self.assertAlmostEqual(0.0, self.difference(C.block('oo'), c2))

    def test_Cia_plus_equal_Aia_minus_three_Bai(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3
        nv = 5

        [Aov_t, a2] = self.build_and_fill("Aov", [no, nv])
        [Bvo_t, b2] = self.build_and_fill("Bvo", [nv, no])
        [Cov_t, c2] = self.build_and_fill("Cov", [no, nv])

        A.block('ov')['pq'] = Aov_t['pq']
        B.block('vo')['pq'] = Bvo_t['pq']
        C.block('ov')['pq'] = Cov_t['pq']

        for i in range(no):
            for a in range(nv):
                c2[i][a] += a2[i][a] - 3.0 * b2[a][i]

        C['ia'] += A['ia'] - 3.0 * B['ai']

        self.assertAlmostEqual(0.0, self.difference(C.block('ov'), c2))

    def test_Dij_equal_Aij_times_Bij_plus_Cij(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,4,5], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])
        D = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "D", ["oo", "ov", "vo", "vv"])

        no = 5

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])
        [Doo_t, d2] = self.build_and_fill("Doo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']
        D.block('oo')['pq'] = Doo_t['pq']

        for i in range(no):
            for j in range(no):
                d2[i][j] = a2[i][j] * (2.0 * b2[i][j] - c2[i][j])

        D['ij'] = A['ij'] * (2.0 * B['ij'] - C['ij'])

        self.assertAlmostEqual(0.0, self.difference(D.block('oo'), d2), places=12)

    def test_Dij_equal_Bij_plus_Cij_times_Aij(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,4,5], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])
        D = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "D", ["oo", "ov", "vo", "vv"])

        no = 5

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])
        [Doo_t, d2] = self.build_and_fill("Doo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']
        D.block('oo')['pq'] = Doo_t['pq']

        for i in range(no):
            for j in range(no):
                d2[i][j] = a2[i][j] * (2.0 * b2[i][j] - c2[i][j])

        D['ij'] = (2.0 * B['ij'] - C['ij']) * A['ij']

        self.assertAlmostEqual(0.0, self.difference(D.block('oo'), d2), places=12)

    def test_Dij_plus_equal_Bij_plus_Cij_times_Aij(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,4,5], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])
        D = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "D", ["oo", "ov", "vo", "vv"])

        no = 5

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])
        [Doo_t, d2] = self.build_and_fill("Doo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']
        D.block('oo')['pq'] = Doo_t['pq']

        for i in range(no):
            for j in range(no):
                d2[i][j] += a2[i][j] * (2.0 * b2[i][j] - c2[i][j])

        D['ij'] += (2.0 * B['ij'] - C['ij']) * A['ij']

        self.assertAlmostEqual(0.0, self.difference(D.block('oo'), d2), places=12)

    def test_Dij_minus_equal_Bij_plus_Cij_times_Aij(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,4,5], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])
        D = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "D", ["oo", "ov", "vo", "vv"])

        no = 5

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])
        [Doo_t, d2] = self.build_and_fill("Doo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']
        D.block('oo')['pq'] = Doo_t['pq']

        for i in range(no):
            for j in range(no):
                d2[i][j] -= a2[i][j] * (2.0 * b2[i][j] - c2[i][j])

        D['ij'] -= (2.0 * B['ij'] - C['ij']) * A['ij']

        self.assertAlmostEqual(0.0, self.difference(D.block('oo'), d2), places=12)

    def test_F_equal_D_times_2g_minus_g(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [6,7,8,9], ambit.SpinType.AlphaSpin)

        F = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "F", ["oo", "ov", "vo", "vv"])
        D = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "D", ["oo", "ov", "vo", "vv"])
        g = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "g", ["oooo", "vvvv"])

        no = 5

        [Foo_t, a2] = self.build_and_fill("Foo", [no, no])
        [Doo_t, b2] = self.build_and_fill("Doo", [no, no])
        [goo_t, c4] = self.build_and_fill("goo", [no, no, no, no])

        F.block('oo')['pq'] = Foo_t['pq']
        D.block('oo')['pq'] = Doo_t['pq']
        g.block('oooo')['pqrs'] = goo_t['pqrs']

        for i in range(no):
            for j in range(no):
                a2[i][j] = 0.0
                for k in range(no):
                    for l in range(no):
                        a2[i][j] += b2[k][l] * (2.0 * c4[i][j][k][l] - c4[i][k][j][l])

        F['i,j'] = D['k,l'] * (2.0 * g['i,j,k,l'] - g['i,k,j,l'])
        F['c,d'] = D['a,b'] * (2.0 * g['a,b,c,d'] - g['a,c,b,d'])

        self.assertAlmostEqual(0.0, self.difference(F.block('oo'), a2), places=12)

    def test_Cij_equal_2_times_Aij_minus_Bij(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']

        for i in range(no):
            for j in range(no):
                c2[i][j] = 2.0 * (a2[i][j] - b2[i][j])

        C['ij'] = 2.0 * (A['ij'] - B['ij'])

        self.assertAlmostEqual(0.0, self.difference(C.block('oo'), c2), places=12)

    def test_Cij_minus_equal_3_times_Aij_minus_2Bij(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']

        for i in range(no):
            for j in range(no):
                c2[i][j] -= 3.0 * (a2[i][j] - 2.0 * b2[i][j])

        C['ij'] -= 3.0 * (A['ij'] - 2.0 * B['ij'])

        self.assertAlmostEqual(0.0, self.difference(C.block('oo'), c2), places=12)

    def test_Cij_equal_negate_Aij_plus_Bij(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 3

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']

        for i in range(no):
            for j in range(no):
                c2[i][j] = - (a2[i][j] + b2[i][j])

        C['ij'] = - (A['ij'] + B['ij'])

        self.assertAlmostEqual(0.0, self.difference(C.block('oo'), c2), places=12)

    def test_dot_product1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9,10,11], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])

        no = 5
        nv = 7

        [Aov_t, a2] = self.build_and_fill("Aov", [no, nv])
        [Bvo_t, b2] = self.build_and_fill("Bvo", [nv, no])

        A.block('ov')['pq'] = Aov_t['pq']
        B.block('vo')['pq'] = Bvo_t['pq']

        c2 = 0.0
        for i in range(no):
            for a in range(nv):
                c2 += a2[i][a] * b2[a][i]

        C = float(A['ia'] * B['ai'])

        self.assertAlmostEqual(0.0, C - c2, places=12)

    @unittest.expectedFailure
    def test_dot_product2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9,10,11], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])

        no = 5
        nv = 7

        [Aov_t, a2] = self.build_and_fill("Aov", [no, nv])
        [Bvo_t, b2] = self.build_and_fill("Bvo", [nv, no])

        A.block('ov')['pq'] = Aov_t['pq']
        B.block('vo')['pq'] = Bvo_t['pq']

        c2 = 0.0
        for i in range(no):
            for a in range(nv):
                c2 += a2[i][a] * b2[a][i]

        C = float(A['ia'] * B['bi'])

        self.assertAlmostEqual(0.0, C - c2, places=12)

    @unittest.expectedFailure
    def test_dot_product3(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9,10,11], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])

        no = 5
        nv = 7

        [Aov_t, a2] = self.build_and_fill("Aov", [no, nv])
        [Bvo_t, b2] = self.build_and_fill("Bvo", [nv, no])

        A.block('ov')['pq'] = Aov_t['pq']
        B.block('vo')['pq'] = Bvo_t['pq']

        c2 = 0.0
        for i in range(no):
            for a in range(nv):
                c2 += a2[i][a] * b2[a][i]

        C = float(A['ia'] * B['aij'])

        self.assertAlmostEqual(0.0, C - c2, places=12)

    def test_dot_product4(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2,4,5], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["oo", "ov", "vo", "vv"])

        no = 5

        [Aoo_t, a2] = self.build_and_fill("Aoo", [no, no])
        [Boo_t, b2] = self.build_and_fill("Boo", [no, no])
        [Coo_t, c2] = self.build_and_fill("Coo", [no, no])

        A.block('oo')['pq'] = Aoo_t['pq']
        B.block('oo')['pq'] = Boo_t['pq']
        C.block('oo')['pq'] = Coo_t['pq']

        d = 0.0
        for i in range(no):
            for j in range(no):
                d += a2[i][j] * (b2[i][j] + c2[i][j])

        D = float(A['ij'] * (B['ij'] + C['ij']))

        self.assertAlmostEqual(0.0, D - d, places=12)

    @unittest.expectedFailure
    def test_contraction_exception1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j,k,l", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)

        A = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "A", ["oo", "ov", "vo", "vv"])
        B = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "B", ["oo", "ov", "vo", "vv"])
        C = ambit.BlockedTensor.build(ambit.TensorType.CoreTensor, "C", ["ov", "vo", "vv"])

        C['ij'] = A['ia'] * B['aj']

if __name__ == '__main__':
    unittest.main()
