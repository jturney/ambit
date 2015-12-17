import unittest
import random
import ambit
import numpy as np
import itertools as it

# Tuneage, will move later
tensor_type = ambit.TensorType.CoreTensor
max_test_rank = 4

# Full test expands considerably on the number of permutations tried
full_test = True

# Sizes
dim_sizes = [9, 6, 3, 3, 3, 3, 3]
dim_inds  = 'ijklmnop'
dim_size_dict = {k:v for k, v in zip(dim_inds, dim_sizes)}

class TestOperatorOverloading(unittest.TestCase):

    def setUp(self):
        ambit.initialize()

    def tearDown(self):
        ambit.finalize()

    def build(self, name, dims, ttype=tensor_type, fill=None):
        # Builds arbitrary Tensors and numpy arrys
        # Accepts a list of integers or a string for dimensions

        if ttype != ambit.TensorType.CoreTensor:
            raise ValueError("Only CoreTensor is currently supported")

        if isinstance(dims, str):
            dims = [dim_size_dict[i] for i in dims]

        T = ambit.Tensor(ttype, name, dims)

        # Fill both N and T
        N = np.asarray(T)
        if fill:
            N.flat[:] = fill
        else:
            N.flat[:] = np.arange(np.prod(dims))

        # Copy numpy array so we no longer share memory
        N = N.copy()

        return [T, N]

    def assert_allclose(self, T, N):
        # Can test arbitrary tensor types once implemented
        if T.dtype == ambit.TensorType.CoreTensor:
            self.assertTrue(np.allclose(T, N))
        else:
            raise ValueError("Assert Allclose: Only CoreTensor is currently supported")

    def test_float_tensor_mul(self):

        for rank in range(1, max_test_rank):
            tmp_dims = dim_sizes[:rank]
            tmp_inds = dim_inds[:rank]

            [aA, nA] = self.build("A", tmp_dims)
            [aC, nC] = self.build("C", tmp_dims)

            aC[tmp_inds] = 2.0 * aA[tmp_inds]
            nC = 2.0 * nA
            self.assert_allclose(aC, nC)

    def test_tensor_tensor_mul(self):
        # Hadamard product

        for rank in range(1, max_test_rank):
            tmp_dims = dim_sizes[:rank]
            tmp_inds = dim_inds[:rank]

            [aA, nA] = self.build("A", tmp_dims)
            [aB, nB] = self.build("B", tmp_dims)
            [aC, nC] = self.build("C", tmp_dims)

            aC[tmp_inds] = aA[tmp_inds] * aB[tmp_inds]
            nC = nA * nB
            self.assert_allclose(aC, nC)

    # Not yet implemented
    #def test_tensor_tensor_imul(self):
    #    # Inplace Hadamard product

    #    for rank in range(1, max_test_rank):
    #        tmp_dims = dim_sizes[:rank]
    #        tmp_inds = dim_inds[:rank]

    #        [aA, nA] = self.build("A", tmp_dims)
    #        [aC, nC] = self.build("C", tmp_dims)

    #        aC[tmp_inds] *= aA[tmp_inds]
    #        nC *= nA
    #        self.assert_allclose(aC, nC)

    def test_tensor_tensor_add(self):

        for rank in range(1, max_test_rank):
            tmp_dims = dim_sizes[:rank]
            tmp_inds = dim_inds[:rank]

            [aA, nA] = self.build("A", tmp_dims)
            [aB, nB] = self.build("B", tmp_dims)
            [aC, nC] = self.build("C", tmp_dims)

            aC[tmp_inds] = aA[tmp_inds] + aB[tmp_inds]
            nC = nA + nB
            self.assert_allclose(aC, nC)

    def test_tensor_tensor_sub(self):

        for rank in range(1, max_test_rank):
            tmp_dims = dim_sizes[:rank]
            tmp_inds = dim_inds[:rank]

            [aA, nA] = self.build("A", tmp_dims)
            [aB, nB] = self.build("B", tmp_dims)
            [aC, nC] = self.build("C", tmp_dims)

            aC[tmp_inds] = aA[tmp_inds] - aB[tmp_inds]
            nC = nA - nB
            self.assert_allclose(aC, nC)

    def test_2d_dot(self):

        ni, nj = 10, 20
        [aA, nA] = self.build("A", [ni, nj])
        [aB, nB] = self.build("B", [nj, ni])
        [aC, nC] = self.build("C", [ni, ni], fill=0)

        aC["ik"] = aA["ij"] * aB["jk"]
        nC = np.dot(nA, nB)
        self.assert_allclose(aC, nC)

    def test_nd_dot_1idx(self):
        # All permutations of nd dot with one index removed
        # Can include all output permutations

        for rank in range(1, max_test_rank):
            perm1 = tuple(it.permutations(dim_inds[:rank]))
            perm2 = tuple(it.permutations(dim_inds[rank - 1:rank * 2 - 1]))
            collapse_ind = dim_inds[rank - 1]

            for comb in it.product(perm1, perm2):
                left_inds = ''.join(comb[0])
                right_inds = ''.join(comb[1])
                [aA, nA] = self.build("A", left_inds)
                [aB, nB] = self.build("B", right_inds)

                ret_inds =  (left_inds + right_inds).replace(collapse_ind, '')
                if full_test:
                    ret_ind_list = it.permutations(ret_inds)
                else:
                    ret_ind_list = [ret_inds]

                for iret_inds in ret_ind_list:
                    iret_inds = ''.join(iret_inds)
                    einsum_string = left_inds + ',' + right_inds + '->' + iret_inds

                    [aC, nC] = self.build("C", iret_inds)
                    aC[iret_inds] = aA[left_inds] * aB[right_inds]
                    np.einsum(einsum_string, nA, nB, out=nC)
                    self.assert_allclose(aC, nC)

    def test_nd_dot_2idx(self):
        # All permutations of nd dot with two index removed
        # Can include all output permutations

        for rank in range(2, max_test_rank):
            perm1 = tuple(it.permutations(dim_inds[:rank]))
            perm2 = tuple(it.permutations(dim_inds[rank - 2:rank * 2 - 2]))
            collapse_ind = dim_inds[rank - 2:rank]

            for comb in it.product(perm1, perm2):
                left_inds = ''.join(comb[0])
                right_inds = ''.join(comb[1])
                [aA, nA] = self.build("A", left_inds)
                [aB, nB] = self.build("B", right_inds)

                ret_inds = (left_inds + right_inds)
                for collapse in collapse_ind:
                    ret_inds = ret_inds.replace(collapse, '')

                if full_test:
                    ret_ind_list = it.permutations(ret_inds)
                else:
                    ret_ind_list = [ret_inds]

                for iret_inds in ret_ind_list:
                    iret_inds = ''.join(iret_inds)
                    einsum_string = left_inds + ',' + right_inds + '->' + iret_inds

                    [aC, nC] = self.build("C", iret_inds)
                    aC[iret_inds] = aA[left_inds] * aB[right_inds]
                    np.einsum(einsum_string, nA, nB, out=nC)
                    self.assert_allclose(aC, nC)

    def test_slices(self):
        [aA, nA] = self.build("A", [10, 10])
        [aB, nB] = self.build("B", [10, 10])

        aA[[3,5], [5,7]] += aB[[2,4], [4,6]]
        nA[3:5, 5:7] += nB[2:4, 4:6]
        self.assert_allclose(aA, nA)

        aA[:, :] += aB[:, :]
        nA[:, :] += nB[:, :]
        self.assert_allclose(aA, nA)

        aA[[3,5], [0,10]] -= aB[[2,4], :]
        nA[3:5, :] -= nB[2:4, :]
        self.assert_allclose(aA, nA)

        aA[5:, [0,10]] -= aB[[0,5], :]
        nA[5:, :] -= nB[:5, :]
        self.assert_allclose(aA, nA)

        aA[3:5, 5:7] = aB[2:4, 4:6]
        nA[3:5, 5:7] = nB[2:4, 4:6]
        self.assert_allclose(aA, nA)

        aA[:5, :] = aB[1:6, :]
        nA[:5, :] = nB[1:6, :]
        self.assert_allclose(aA, nA)




if __name__ == '__main__':
    unittest.main()

