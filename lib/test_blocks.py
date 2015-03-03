import unittest
import random
import numpy as np
import ambit

class TestBlocks(unittest.TestCase):

    def build_and_fill2(self, name, dims):
        T = ambit.Tensor(ambit.TensorType.kCore, name, dims)
        N = [[0 for x in range(dims[1])] for x in range(dims[0])]

        data = T.tensor.data()
        for r in range(dims[0]):
            for c in range(dims[1]):
                value = random.random()
                data[r * dims[0] + c] = value
                N[r][c] = value

        return [T, N]

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
        A = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T", ["oo", "vv"])
        # A.printf()

    def test_block_creation2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", ["o", "v"])
        A = ambit.BlockedTensor.build(ambit.TensorType.kCore, "F", ["gg"])
        B = ambit.BlockedTensor.build(ambit.TensorType.kCore, "V", ["gggg"])

    def test_block_creation3(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("c", "m,n", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("a", "u,v", [3,4], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("h", "i,j,k,l", ["c", "a"])
        ambit.BlockedTensor.add_composite_mo_space("p", "a,b,c,d", ["a", "v"])
        ambit.BlockedTensor.build(ambit.TensorType.kCore, "T1", ["hp"])
        ambit.BlockedTensor.build(ambit.TensorType.kCore, "T2", ["hhpp"])

    @unittest.expectedFailure
    def test_block_creation_bad_rank(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.build(ambit.TensorType.kCore, "T", ["oo", "ovv"])

    def test_block_norm_1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T2", ["aavv"])
        T2.set(0.5)
        self.assertAlmostEqual(112.5, T2.norm(1), places=12)

    def test_block_norm_2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T2", ["aavv"])
        T2.set(0.5)
        self.assertAlmostEqual(7.5, T2.norm(2), places=12)

    def test_block_norm_3(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T2", ["aavv"])
        T2.set(0.5)
        self.assertAlmostEqual(0.5, T2.norm(0), places=12)

    def test_block_zero(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T2", ["aavv"])
        T2.set(0.5)
        T2.zero()
        self.assertAlmostEqual(0.0, T2.norm(2), places=12)

    def test_block_scale(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("a", "u,v", [2,3,4], ambit.SpinType.NoSpin)
        ambit.BlockedTensor.add_mo_space("v", "e,f", [5,6,7,8,9], ambit.SpinType.NoSpin)
        T2 = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T2", ["aavv"])
        T2.set(2.0)
        T2.scale(0.25)
        self.assertAlmostEqual(7.5, T2.norm(2), places=12)

    def test_block_labels1(self):
        pass

    def test_block_retrieve_block1(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        T = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T", ["oo", "vv"])
        T.block("oo")

    @unittest.expectedFailure
    def test_block_retrieve_block2(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_composite_mo_space("g", "p,q,r,s", {"o", "v"})
        T = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T", {"oo", "vv"})
        T.block("og")

    @unittest.expectedFailure
    def test_block_retrieve_block3(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        T = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T", ["oo", "vv"])
        T.block("ov")

    @unittest.expectedFailure
    def test_block_retrieve_block4(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        T = ambit.BlockedTensor.build(ambit.TensorType.kCore, "T", ["oo", "vv"])
        T.block("")

    def test_block_iterator1(self):
        pass

    def test_Cij_equal_Aji(self):
        ambit.BlockedTensor.reset_mo_space()
        ambit.BlockedTensor.add_mo_space("o", "i,j", [0,1,2], ambit.SpinType.AlphaSpin)
        ambit.BlockedTensor.add_mo_space("v", "a,b,c,d", [5,6,7,8,9], ambit.SpinType.AlphaSpin)
        A = ambit.BlockedTensor.build(ambit.TensorType.kCore, "A", ["oo", "vv", "ov", "vo"])
        C = ambit.BlockedTensor.build(ambit.TensorType.kCore, "C", ["oo", "vv", "ov", "vo"])

        Aoo = A.block("oo")
        Coo = C.block("oo")

        no = 3
        nv = 5

        [Aoo_t, a2] = self.build_and_fill2("A", [no, no])
        [Coo_t, c2] = self.build_and_fill2("C", [no, no])

        print(Aoo)
        print(Aoo_t)

        Aoo["ij"] = Aoo_t["ij"]
        Coo["ij"] = Coo_t["ij"]

        C["ij"] = A["ji"]

        for i in range(no):
            for j in range(no):
                c2[i][j] = a2[i][j]


if __name__ == '__main__':
    unittest.main()
