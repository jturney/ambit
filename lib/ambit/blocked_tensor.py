from . import pyambit
from . import tensor_wrapper
import math
import copy
import numbers


class SpinType:
    AlphaSpin = 1
    BetaSpin = 2
    NoSpin = 3


class MOSpace:
    def __init__(self, name, mo_indices, mos, spin):
        self.name = str(name)
        self.mo_indices = pyambit.Indices.split(mo_indices)
        self.mos = mos

        if len(self.name) == 0:
            raise RuntimeError("MOSpace: No name provided to MO space")

        if len(mo_indices) == 0:
            raise RuntimeError("MOSpace: No MO indices provided.")

        if len(mos) == 0:
            raise RuntimeError("MOSpace: No MOs provided.")

        if SpinType.AlphaSpin <= spin <= SpinType.NoSpin:
            self.spin = spin
        else:
            raise TypeError("Value of spin is invalid.")

    def dim(self):
        return len(self.mos)

    def __str__(self):
        msg = "\n  Orbital Space \"%s\"\n  MO Indices: {%s}\n  MO List: (%s)\n" % (
        self.name, ','.join(map(str, self.mo_indices)), ','.join(map(str, self.mos)))
        return msg

class LabeledBlockedTensorProduct:

    def __init__(self, left, right):
        self.btensors = []
        self.btensors.append(left)
        self.btensors.append(right)

    def __mul__(self, other):
        if isinstance(other, LabeledBlockedTensor):
            self.btensors.append(other)
            return self

class LabeledBlockedTensorAddition:
    pass

class LabeledBlockedTensorDistributive:
    pass

class LabeledBlockedTensor:
    def __init__(self, T, indices, factor=1.0):
        self.btensor = T
        self.indices = indices
        self.indices_split = pyambit.Indices.split(indices)
        self.factor = factor

    def add(self, rhs, alpha, beta):
        rhs_keys = rhs.label_to_block_keys()

        perm = pyambit.Indices.permutation_order(pyambit.Indices.split(self.indices),
                                                 pyambit.Indices.split(rhs.indices))

        for rhs_key in rhs_keys:
            lhs_key = ""
            for p in perm:
                lhs_key += rhs_key[p]

            # Grab the raw tensors
            LHS = self.btensor.block(lhs_key)
            RHS = rhs.btensor.block(rhs_key)

            # Need to protect against self assignment
            # Need to protect against different ranks
            LHS.permute(RHS, self.indices_split, rhs.indices_split, alpha=alpha * rhs.factor, beta=beta)

    def contract(self, rhs, zero_result, add):
        if isinstance(rhs, LabeledBlockedTensorProduct):
            unique_indices = []
            for term in rhs.btensors:
                for index in term.indices:
                    unique_indices.append(index)

            unique_indices.sort()
            # print(unique_indices)

            unique_indices = list(set(unique_indices))
            unique_indices.sort()

            # print(unique_indices)

            unique_indices_key = BlockedTensor.label_to_block_keys(unique_indices)

            # print("unique_indices_key: " + str(unique_indices_key))

            index_map = {}
            k = 0
            for index in unique_indices:
                index_map[index] = k
                k += 1

            # print("index_map: " + str(index_map))

            if zero_result == True:
                for uik in unique_indices_key:
                    # print("uik: " + str(uik))
                    result_key = ""
                    for index in self.indices:
                        result_key += uik[index_map[index]]
                    # print("result_key: " + str(result_key))
                    self.btensor.block(result_key).zero()

            # Setup and perform contractions
            for uik in unique_indices_key:
                result_key = ""
                for index in self.indices:
                    result_key += uik[index_map[index]]
                result = tensor_wrapper.LabeledTensor(self.btensor.block(result_key).tensor, self.indices, self.factor)

                prod = tensor_wrapper.LabeledTensorProduct(None, None)
                for lbt in rhs.btensors:
                    term_key = ""
                    for index in lbt.indices:
                        term_key += uik[index_map[index]]
                    term = tensor_wrapper.LabeledTensor(lbt.btensor.block(term_key).tensor, lbt.indices, lbt.factor)
                    prod.tensors.append(term)

                if add == True:
                    result += prod
                else:
                    result -= prod

        else:
            raise RuntimeError("LabeledBlockedTensor.contract: Unexpected type for rhs: " + type(rhs))

    def label_to_block_keys(self):
        return self.btensor.label_to_block_keys(self.indices)

    def __neg__(self):
        self.factor *= -1.0
        return self

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other
            return self
        elif isinstance(other, LabeledBlockedTensorAddition):
            return LabeledBlockedTensorDistributive(self, other)
        else:
            return LabeledBlockedTensorProduct(self, other)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other
            return self

    def __iadd__(self, other):
        if isinstance(other, LabeledBlockedTensor):
            self.add(other, 1.0, 1.0)
            return None
        elif isinstance(other, LabeledBlockedTensorProduct):
            self.contract(other, False, True)
            return None

    def __imul__(self, other):
        if isinstance(other, numbers.Number):
            keys = self.label_to_block_keys()

            for key in keys:
                self.btensor.block(key).scale(other)

    def __isub__(self, other):
        if isinstance(other, LabeledBlockedTensor):
            self.add(other, -1.0, 1.0)
            return None
        elif isinstance(other, LabeledBlockedTensorProduct):
            self.contract(other, False, False)
            return None

    def __itruediv__(self, other):
        if isinstance(other, numbers.Number):
            keys = self.label_to_block_keys()

            for key in keys:
                self.btensor.block(key).scale(1.0 / other)

class BlockedTensor:
    mo_spaces = []
    name_to_mo_space = {}
    composite_name_to_mo_spaces = {}
    index_to_mo_spaces = {}

    @staticmethod
    def add_mo_space(name, mo_indices, mos, spin):

        if str(name) in BlockedTensor.name_to_mo_space:
            raise RuntimeError("The MO space \"%s\" is already defined." % name)

        mo_space_idx = len(BlockedTensor.mo_spaces)

        ms = MOSpace(name, mo_indices, mos, spin)

        # Add the MOSpace object
        BlockedTensor.mo_spaces.append(ms)

        # Link the name to the mo_space vector
        BlockedTensor.name_to_mo_space[name] = mo_space_idx

        # Link the composite name to the mo_space vector
        BlockedTensor.composite_name_to_mo_spaces[name] = [mo_space_idx]

        # Link the indices to the mo_space
        for mo_index in pyambit.Indices.split(mo_indices):
            if mo_index in BlockedTensor.index_to_mo_spaces:
                raise RuntimeError("The MO index \"%s\" is already defined." % mo_index)
            else:
                BlockedTensor.index_to_mo_spaces[mo_index] = [mo_space_idx]

    @staticmethod
    def add_composite_mo_space(name, mo_indices, subspaces):

        if len(str(name)) == 0:
            raise RuntimeError("Attempting to add compose MO space with no name.")

        if len(mo_indices) == 0:
            raise RuntimeError("No MO indices provided to compose MO space.")

        if name in BlockedTensor.name_to_mo_space:
            raise RuntimeError("The MO space \"%s\" is already defined." % name)

        simple_spaces = []
        for subspace in subspaces:
            if subspace in BlockedTensor.name_to_mo_space:
                simple_spaces.append(BlockedTensor.name_to_mo_space[subspace])
            else:
                raise RuntimeError("The simple MO space \"%s\" is not defined." % subspace)
        BlockedTensor.composite_name_to_mo_spaces[name] = simple_spaces

        # Link the indices to the mo_space
        for mo_index in pyambit.Indices.split(mo_indices):
            if mo_index in BlockedTensor.index_to_mo_spaces:
                raise RuntimeError("The MO index \"%s\" is already defined." % mo_index)
            else:
                BlockedTensor.index_to_mo_spaces[mo_index] = simple_spaces

    @staticmethod
    def print_mo_spaces():
        print("\n  List of Molecular Orbital Spaces:")
        for mo_space in BlockedTensor.mo_spaces:
            print(mo_space)

    @staticmethod
    def reset_mo_space():
        BlockedTensor.mo_spaces = []
        BlockedTensor.name_to_mo_space = {}
        BlockedTensor.composite_name_to_mo_spaces = {}
        BlockedTensor.index_to_mo_spaces = {}

    def __init__(self):
        self.rank = 0
        self.name = "(empty)"
        self.blocks = {}

    @staticmethod
    def build(type, name, blocks):
        newObject = BlockedTensor()

        newObject.name = str(name)
        newObject.rank = 0

        tensor_blocks = []

        # This algorithm takes a vector of strings that define the blocks of this tensor and unpacks them.
        # This may require taking composite spaces ("G" = {"O","V"}) and expanding the string
        # "G,G" -> {"O,O" , "O,V" , "V,O" , "V,V"}.
        # The way we proceed is by forming partial strings that we keep expanding as we
        # process all the indices.

        # Transform composite indices into simple indices
        for this_block in blocks:
            final_blocks = []

            this_block_vec = pyambit.Indices.split(this_block)

            # Loop over indices of this block
            for mo_space_name in this_block_vec:
                partial_blocks = []
                # How does this MO space name map to the MOSpace object contained in mo_spaces ? (e.g. "G" -> {0,1})
                for mo_space_idx in BlockedTensor.composite_name_to_mo_spaces[mo_space_name]:
                    # Special case
                    if len(final_blocks) == 0:
                        partial_blocks.append([mo_space_idx])
                    else:
                        # Add each primitive set to all the partial block labels
                        for block in final_blocks:
                            new_block = copy.deepcopy(block)
                            new_block.append(mo_space_idx)
                            partial_blocks.append(new_block)

                final_blocks = partial_blocks

            for block in final_blocks:
                tensor_blocks.append(block)

        # Create the blocks
        for this_block in tensor_blocks:
            # Grab the dims
            dims = []

            for ms in this_block:
                dims.append(BlockedTensor.mo_spaces[ms].dim())

            # Grab the orbital spaces names
            mo_names = ""
            for ms in this_block:
                mo_names += BlockedTensor.mo_spaces[ms].name

            newObject.blocks[mo_names] = tensor_wrapper.Tensor.build(type=type, name=name + "[" + mo_names + "]",
                                                                     dims=dims)

            # Set or check the rank
            if newObject.rank > 0:
                if newObject.rank != len(this_block):
                    raise RuntimeError("Attempting to create the BlockedTensor \"" + name + "\" with nonunique rank.")
            else:
                newObject.rank = len(this_block)

        return newObject

    def numblocks(self):
        return len(self.blocks)

    def indices_to_key(self, indices):
        key = []
        for index in pyambit.Indices.split(indices):
            if index in BlockedTensor.name_to_mo_space:
                key.append(BlockedTensor.name_to_mo_space[index])
            else:
                raise RuntimeError("The index " + index + " does not identify a unique space (indices_to_key).")

        return key

    def is_block(self, indices):
        if isinstance(indices, str):
            return self.is_block(self.indices_to_key(indices))
        else:
            mo_names = ""
            for k in indices:
                mo_names += BlockedTensor.mo_spaces[k].name

            return mo_names in self.blocks

    def block(self, indices):
        if isinstance(indices, str):
            key = []
            for index in pyambit.Indices.split(indices):
                if index in BlockedTensor.name_to_mo_space:
                    key.append(BlockedTensor.name_to_mo_space[index])
                else:
                    raise RuntimeError(
                        "Cannot retrieve block " + indices + " of tensor " + self.name + ". The index " + index + " does not identify a unique space.")
            return self.block(key)

        else:
            mo_names = ""
            for k in indices:
                mo_names += BlockedTensor.mo_spaces[k].name

            if not self.is_block(mo_names):
                msg = ""
                for k in indices:
                    msg += str(k) + "(" + BlockedTensor.mo_space[k].name + ")"
                raise RuntimeError("Block \"" + msg + "\" is not contained in tensor " + self.name)

            return self.blocks[mo_names]

    def norm(self, type):

        if type == 0:
            val = 0.0
            for key in self.blocks:
                val = max(val, abs(self.blocks[key].norm(type)))
            return val

        elif type == 1:
            val = 0.0
            for key in self.blocks:
                val += abs(self.blocks[key].norm(type))
            return val

        elif type == 2:
            val = 0.0
            for key in self.blocks:
                val += pow(self.blocks[key].norm(type), 2.0)
            return math.sqrt(val)

        else:
            raise RuntimeError("Norm must be 0 (infty-norm), 1 (1-norm), or 2 (2-norm)")

        return 0.0

    def zero(self):
        for key in self.blocks:
            self.blocks[key].zero()

    def scale(self, beta):
        for key in self.blocks:
            self.blocks[key].scale(beta)

    @staticmethod
    def label_to_block_keys(indices):

        # This function takes in the labels used to form a LabeledBlockedTensor and returns the keys
        # to access the corresponding blocks.
        # For example, suppose that indices "i,j,k" are reserved for occupied orbitals ("o") and that index "p"
        # belongs to the composite space of occupied and virtual orbitals ("o" + "v").
        # Then if this function is called with {"i","j","k","p"} it will return the vectors
        # {0,0,0,0} and {0,0,0,1}, which stand for the "oooo" and "ooov" blocks, respectively.
        # The way we proceed is by forming partial vectors that we keep expanding as we
        # process all the indices.

        if not isinstance(indices, list):
            indices = pyambit.Indices.split(indices)

        final_blocks = []

        # Loop over indices of this block
        for index in indices:
            partial_blocks = []

            # How does this MO space name map to the MOSpace objects contained in mo_spaces
            if index in BlockedTensor.index_to_mo_spaces:
                for mo_space_idx in BlockedTensor.index_to_mo_spaces[index]:
                    # Special case
                    if len(final_blocks) == 0:
                        partial_blocks.append([mo_space_idx])
                    else:
                        # Add each primitive set to all the partial block labels
                        for block in final_blocks:
                            new_block = copy.deepcopy(block)
                            new_block.append(mo_space_idx)
                            partial_blocks.append(new_block)

            else:
                raise RuntimeError("Index \"" + index + "\" is not defined.")

            final_blocks = partial_blocks

        # Grab the orbital spaces names
        mo_names = []
        for block in final_blocks:
            mo_name = ""
            for ms in block:
                mo_name += BlockedTensor.mo_spaces[ms].name
            mo_names.append(mo_name)

        return mo_names

    def printf(self):
        print("  ## Blocked Tensor " + self.name + " ##\n")
        print("  Number of blocks = %d" % (self.numblocks()))

        for key in self.blocks:
            self.blocks[key].printf()

    def set(self, gamma):
        for key in self.blocks:
            data = self.blocks[key].data()
            for i, p in enumerate(data):
                data[i] = gamma

    def __getitem__(self, indices):
        return LabeledBlockedTensor(self, indices)

    def __setitem__(self, indices_str, rhs):
        indices = pyambit.Indices.split(str(indices_str))

        if isinstance(rhs, LabeledBlockedTensor):
            me = LabeledBlockedTensor(self, indices_str)
            me.add(rhs, 1.0, 0.0)

        elif isinstance(rhs, LabeledBlockedTensorProduct):
            me = LabeledBlockedTensor(self, indices_str)
            me.contract(rhs, True, True)
