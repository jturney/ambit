/*
 * @BEGIN LICENSE
 *
 * ambit: C++ library for the implementation of tensor product calculations
 *        through a clean, concise user interface.
 *
 * Copyright (c) 2014-2017 Ambit developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of ambit.
 *
 * Ambit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Ambit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <numeric>
#include <ambit/blocked_tensor.h>
#include <tensor/indices.h>

namespace ambit
{

// Static members of BlockedTensor
std::vector<MOSpace> BlockedTensor::mo_spaces_;
std::map<std::string, size_t> BlockedTensor::name_to_mo_space_;
std::map<std::string, std::vector<size_t>>
    BlockedTensor::composite_name_to_mo_spaces_;
std::map<std::string, std::vector<size_t>> BlockedTensor::index_to_mo_spaces_;

bool BlockedTensor::expert_mode_ = false;

MOSpace::MOSpace(const std::string &name, const std::string &mo_indices,
                 std::vector<size_t> mos, SpinType spin)
    : name_(name), mo_indices_(indices::split(mo_indices)), mos_(mos),
      spin_(mos.size(), spin)
{
}

MOSpace::MOSpace(const std::string &name, const std::string &mo_indices,
                 std::vector<std::pair<size_t, SpinType>> mos_spin)
    : name_(name), mo_indices_(indices::split(mo_indices))
{
    for (const auto &p_s : mos_spin)
    {
        mos_.push_back(p_s.first);
        spin_.push_back(p_s.second);
    }
}

void MOSpace::print()
{
    std::vector<std::string> mo_list;
    for (size_t i : mos_)
    {
        mo_list.push_back(std::to_string(i));
    }
    printf("\n  Orbital Space \"%s\"\n  MO Indices: {%s}\n  MO List: (%s)\n",
           name_.c_str(), indices::to_string(mo_indices_).c_str(),
           indices::to_string(mo_list).c_str());
}

void BlockedTensor::add_mo_space(const std::string &name,
                                 const std::string &mo_indices,
                                 std::vector<size_t> mos, SpinType spin)
{
    if (name.size() == 0)
    {
        throw std::runtime_error("Empty name given to orbital space.");
    }
    if (mo_indices.size() == 0)
    {
        throw std::runtime_error(
            "No MO indices were specified for the MO space \"" + name + "\"");
    }
    if (name_to_mo_space_.count(name) != 0)
    {
        throw std::runtime_error("The MO space \"" + name +
                                 "\" is already defined.");
    }

    size_t mo_space_idx = mo_spaces_.size();

    MOSpace ms(name, mo_indices, mos, spin);
    // Add the MOSpace object
    mo_spaces_.push_back(ms);

    // Link the name to the mo_space_ vector
    name_to_mo_space_[name] = mo_space_idx;

    // Link the composite name to the mo_space_ vector
    composite_name_to_mo_spaces_[name] = {mo_space_idx};

    // Link the indices to the mo_space_
    for (const std::string &mo_index : indices::split(mo_indices))
    {
        if (index_to_mo_spaces_.count(mo_index) == 0)
        {
            index_to_mo_spaces_[mo_index] = {mo_space_idx};
        }
        else
        {
            throw std::runtime_error("The MO index \"" + mo_index +
                                     "\" is already defined.");
        }
    }
}

void BlockedTensor::add_mo_space(
    const std::string &name, const std::string &mo_indices,
    std::vector<std::pair<size_t, SpinType>> mo_spin)
{
    if (name.size() == 0)
    {
        throw std::runtime_error("Empty name given to orbital space.");
    }
    if (mo_indices.size() == 0)
    {
        throw std::runtime_error(
            "No MO indices were specified for the MO space \"" + name + "\"");
    }
    if (name_to_mo_space_.count(name) != 0)
    {
        throw std::runtime_error("The MO space \"" + name +
                                 "\" is already defined.");
    }

    size_t mo_space_idx = mo_spaces_.size();

    MOSpace ms(name, mo_indices, mo_spin);
    // Add the MOSpace object
    mo_spaces_.push_back(ms);

    // Link the name to the mo_space_ vector
    name_to_mo_space_[name] = mo_space_idx;

    // Link the composite name to the mo_space_ vector
    composite_name_to_mo_spaces_[name] = {mo_space_idx};

    // Link the indices to the mo_space_
    for (const std::string &mo_index : indices::split(mo_indices))
    {
        if (index_to_mo_spaces_.count(mo_index) == 0)
        {
            index_to_mo_spaces_[mo_index] = {mo_space_idx};
        }
        else
        {
            throw std::runtime_error("The MO index \"" + mo_index +
                                     "\" is already defined.");
        }
    }
}

void BlockedTensor::add_composite_mo_space(
    const std::string &name, const std::string &mo_indices,
    const std::vector<std::string> &subspaces)
{
    if (name.size() == 0)
    {
        throw std::runtime_error(
            "Empty name given to composite orbital space.");
    }
    if (mo_indices.size() == 0)
    {
        throw std::runtime_error(
            "No MO indices were specified for the composite MO space \"" +
            name + "\"");
    }
    if (name_to_mo_space_.count(name) != 0)
    {
        throw std::runtime_error("The MO space \"" + name +
                                 "\" is already defined.");
    }

    std::vector<size_t> simple_spaces;
    for (std::string subspace : subspaces)
    {
        // Is this simple MO space in our list of spaces?
        if (name_to_mo_space_.count(subspace) == 0)
        {
            throw std::runtime_error("The simple MO space \"" + subspace +
                                     "\" is not defined.");
        }
        else
        {
            simple_spaces.push_back(name_to_mo_space_[subspace]);
        }
    }
    composite_name_to_mo_spaces_[name] = simple_spaces;

    // Link the indices to the mo_space_
    for (const std::string &mo_index : indices::split(mo_indices))
    {
        if (index_to_mo_spaces_.count(mo_index) == 0)
        {
            index_to_mo_spaces_[mo_index] = simple_spaces;
        }
        else
        {
            throw std::runtime_error("The MO index \"" + mo_index +
                                     "\" is already defined.");
        }
    }
}

void BlockedTensor::print_mo_spaces()
{
    printf("\n  List of Molecular Orbital Spaces:");
    for (size_t ms = 0; ms < mo_spaces_.size(); ++ms)
    {
        mo_spaces_[ms].print();
    }
}

void BlockedTensor::reset_mo_spaces()
{
    mo_spaces_.clear();
    name_to_mo_space_.clear();
    composite_name_to_mo_spaces_.clear();
    index_to_mo_spaces_.clear();
}

BlockedTensor::BlockedTensor() : rank_(0) {}

BlockedTensor BlockedTensor::build(TensorType type, const std::string &name,
                                   const std::vector<std::string> &blocks)
{
    BlockedTensor newObject;

    newObject.set_name(name);
    newObject.rank_ = 0;

    std::vector<std::vector<size_t>> tensor_blocks;

    // This algorithm takes a vector of strings that define the blocks of this
    // tensor and unpacks them.
    // This may require taking composite spaces ("G" = {"O","V"}) and expanding
    // the string
    // "G,G" -> {"O,O" , "O,V" , "V,O" , "V,V"}.
    // The way we proceed is by forming partial strings that we keep expanding
    // as we
    // process all the indices.

    // Transform composite indices into simple indices
    for (const std::string &this_block : blocks)
    {
        std::vector<std::vector<size_t>> final_blocks;

        std::vector<std::string> this_block_vec = indices::split(this_block);
        // Loop over indices of this block
        for (std::string mo_space_name : this_block_vec)
        {
            std::vector<std::vector<size_t>> partial_blocks;
            // How does this MO space name map to the MOSpace objects contained
            // in mo_spaces_? (e.g. "G" -> {0,1})
            for (size_t mo_space_idx :
                 composite_name_to_mo_spaces_[mo_space_name])
            {
                // Special case
                if (final_blocks.size() == 0)
                {
                    partial_blocks.push_back({mo_space_idx});
                }
                else
                {
                    // Add each this primitive set to all the partial block
                    // labels
                    for (std::vector<size_t> &block : final_blocks)
                    {
                        std::vector<size_t> new_block(block);
                        new_block.push_back(mo_space_idx);
                        partial_blocks.push_back(new_block);
                    }
                }
            }
            final_blocks = partial_blocks;
        }
        for (std::vector<size_t> &block : final_blocks)
            tensor_blocks.push_back(block);
    }

    // Create the blocks
    for (std::vector<size_t> &this_block : tensor_blocks)
    {
        // Grab the dims
        std::vector<size_t> dims;
        for (size_t ms : this_block)
        {
            size_t dim = mo_spaces_[ms].dim();
            dims.push_back(dim);
        }
        // Grab the orbital spaces names
        std::string block_label;
        for (size_t ms : this_block)
        {
            block_label += mo_spaces_[ms].name();
        }
        newObject.blocks_[this_block] =
            Tensor::build(type, name + "[" + block_label + "]", dims);
        newObject.block_labels_.push_back(block_label);

        // Set or check the rank
        if (newObject.rank_ > 0)
        {
            if (newObject.rank_ != this_block.size())
            {
                throw std::runtime_error(
                    "Attempting to create the BlockedTensor \"" + name +
                    "\" with nonunique rank.");
            }
        }
        else
        {
            newObject.rank_ = this_block.size();
        }
    }

    //    newObject.print(stdout);
    return newObject;
}

size_t BlockedTensor::numblocks() const { return blocks_.size(); }

std::string BlockedTensor::name() const { return name_; }

size_t BlockedTensor::rank() const { return rank_; }

void BlockedTensor::set_name(const std::string &name) { name_ = name; }

std::vector<size_t> BlockedTensor::indices_to_key(const std::string &indices)
{
    std::vector<size_t> key;
    for (const std::string &index : indices::split(indices))
    {
        if (name_to_mo_space_.count(index) != 0)
        {
            key.push_back(name_to_mo_space_[index]);
        }
        else
        {
            throw std::runtime_error(
                "The index " + index +
                " does not indentify a unique space (indices_to_key).");
        }
    }
    return key;
}

bool BlockedTensor::is_block(const std::string &indices) const
{
    return is_block(indices_to_key(indices));
}

bool BlockedTensor::is_block(const std::vector<size_t> &key) const
{
    return (blocks_.count(key) != 0);
}

Tensor BlockedTensor::block(const std::string &indices)
{
    std::vector<size_t> key;
    for (const std::string &index : indices::split(indices))
    {
        if (name_to_mo_space_.count(index) != 0)
        {
            key.push_back(name_to_mo_space_[index]);
        }
        else
        {
            throw std::runtime_error(
                "Cannot retrieve block " + indices + " of tensor " + name() +
                ". The index " + index + " does not indentify a unique space");
        }
    }
    return block(key);
}

Tensor BlockedTensor::block(const std::vector<size_t> &key)
{
    if (!is_block(key))
    {
        std::string labels;
        for (size_t k : key)
        {
            labels += mo_space(k).name();
        }
        throw std::runtime_error("Tensor " + name() +
                                 " does not contain block \"" + labels + "\"");
    }
    return blocks_.at(key);
}

const Tensor BlockedTensor::block(const std::vector<size_t> &key) const
{
    if (!is_block(key))
    {
        std::string labels;
        for (size_t k : key)
        {
            labels += mo_space(k).name();
        }
        throw std::runtime_error("Tensor " + name() +
                                 " does not contain block \"" + labels + "\"");
    }
    return blocks_.at(key);
}

double BlockedTensor::norm(int type) const
{
    if (type == 0)
    {
        double val = 0.0;
        for (auto block_tensor : blocks_)
        {
            val = std::max(val, std::fabs(block_tensor.second.norm(type)));
        }
        return val;
    }
    else if (type == 1)
    {
        double val = 0.0;
        for (auto block_tensor : blocks_)
        {
            val += std::fabs(block_tensor.second.norm(type));
        }
        return val;
    }
    else if (type == 2)
    {
        double val = 0.0;
        for (auto block_tensor : blocks_)
        {
            val += std::pow(block_tensor.second.norm(type), 2.0);
        }
        return std::sqrt(val);
    }
    else
    {
        throw std::runtime_error(
            "Norm must be 0 (infty-norm), 1 (1-norm), or 2 (2-norm)");
    }
    return 0.0;
}

void BlockedTensor::zero()
{
    for (auto block_tensor : blocks_)
    {
        block_tensor.second.zero();
    }
}

void BlockedTensor::scale(double beta)
{
    for (auto block_tensor : blocks_)
    {
        block_tensor.second.scale(beta);
    }
}

void BlockedTensor::set(double gamma)
{
    for (auto block_tensor : blocks_)
    {
        block_tensor.second.set(gamma);
    }
}

bool BlockedTensor::operator==(const BlockedTensor &other) const
{
    bool same = false;
    for (auto block_tensor : blocks_)
    {
        const std::vector<size_t> &key = block_tensor.first;
        if (other.is_block(key))
        {
            if (block_tensor.second == other.block(key))
            {
                same = true;
            }
        }
    }
    return same;
}

bool BlockedTensor::operator!=(const BlockedTensor &other) const
{
    return not(*this == other);
}

// void BlockedTensor::copy(const BlockedTensor& other)
//{
//    blocks_.clear();
//    for (auto key_tensor : other.blocks_){
//        Tensor T;
//        T.copy(key_tensor.second);
//        blocks_[key_tensor.first] = T;
//    }
//}(const std::vector<size_t>&,const std::vector<SpinType>&, double&)

void BlockedTensor::iterate(
    const std::function<void(const std::vector<size_t> &,
                             const std::vector<SpinType> &, double &)> &func)
{
    for (auto key_tensor : blocks_)
    {
        const std::vector<size_t> &key = key_tensor.first;

        // Assemble the map from the block indices to the MO indices

        size_t rank = key_tensor.second.rank();
        std::vector<size_t> mo(rank);
        std::vector<SpinType> spin(rank);

        std::vector<std::vector<size_t>> index_to_mo;
        std::vector<std::vector<SpinType>> index_to_spin;
        for (size_t k : key)
        {
            index_to_mo.push_back(mo_spaces_[k].mos());
            index_to_spin.push_back(mo_spaces_[k].spin());
        }

        // Call iterate on this tensor block
        key_tensor.second.iterate(
            [&](const std::vector<size_t> &indices, double &value)
            {
                for (size_t n = 0; n < rank; ++n)
                {
                    mo[n] = index_to_mo[n][indices[n]];
                    spin[n] = index_to_spin[n][indices[n]];
                }
                func(mo, spin, value);
            });
    }
}

void BlockedTensor::citerate(
    const std::function<void(const std::vector<size_t> &,
                             const std::vector<SpinType> &, const double &)>
        &func) const
{
    for (const auto key_tensor : blocks_)
    {
        const std::vector<size_t> &key = key_tensor.first;

        // Assemble the map from the block indices to the MO indices

        size_t rank = key_tensor.second.rank();
        std::vector<size_t> mo(rank);
        std::vector<SpinType> spin(rank);

        std::vector<std::vector<size_t>> index_to_mo;
        std::vector<std::vector<SpinType>> index_to_spin;
        for (size_t k : key)
        {
            index_to_mo.push_back(mo_spaces_[k].mos());
            index_to_spin.push_back(mo_spaces_[k].spin());
        }

        // Call iterate on this tensor block
        key_tensor.second.citerate(
            [&](const std::vector<size_t> &indices, const double &value)
            {
                for (size_t n = 0; n < rank; ++n)
                {
                    mo[n] = index_to_mo[n][indices[n]];
                    spin[n] = index_to_spin[n][indices[n]];
                }
                func(mo, spin, value);
            });
    }
}

void BlockedTensor::print(FILE *fh, bool level, std::string const &format,
                          int maxcols) const
{
    fprintf(fh, "  ## Blocked Tensor %s ##\n\n", name().c_str());
    fprintf(fh, "  Number of blocks = %zu\n", numblocks());
    for (auto kv : blocks_)
    {
        fprintf(fh, "\n");
        kv.second.print(fh, level, format, maxcols);
    }
}

LabeledBlockedTensor BlockedTensor::operator()(const std::string &indices)
{
    return LabeledBlockedTensor(*this, indices::split(indices));
}

LabeledBlockedTensor BlockedTensor::operator[](const std::string &indices)
{
    return LabeledBlockedTensor(*this, indices::split(indices));
}

std::vector<std::vector<size_t>>
BlockedTensor::label_to_block_keys(const std::vector<std::string> &indices)
{
    // This function takes in the labels used to form a LabeledBlockedTensor and
    // returns the keys
    // to access the corresponding blocks.
    // For example, suppose that indices "i,j,k" are reserved for occupied
    // orbitals ("o") and that index "p"
    // belongs to the composite space of occupied and virtual orbitals ("o" +
    // "v").
    // Then if this function is called with {"i","j","k","p"} it will return the
    // vectors
    // {0,0,0,0} and {0,0,0,1}, which stand for the "oooo" and "ooov" blocks,
    // respectively.
    // The way we proceed is by forming partial vectors that we keep expanding
    // as we
    // process all the indices.

    std::vector<std::vector<size_t>> final_blocks;

    // Loop over indices of this block
    for (const std::string &index : indices)
    {
        std::vector<std::vector<size_t>> partial_blocks;
        // How does this MO space name map to the MOSpace objects contained in
        // mo_spaces_? (e.g. "G" -> {0,1})
        if (index_to_mo_spaces_.count(index) != 0)
        {
            for (size_t mo_space_idx : index_to_mo_spaces_[index])
            {
                // Special case
                if (final_blocks.size() == 0)
                {
                    partial_blocks.push_back({mo_space_idx});
                }
                else
                {
                    // Add each this primitive set to all the partial block
                    // labels
                    for (std::vector<size_t> &block : final_blocks)
                    {
                        std::vector<size_t> new_block(block);
                        new_block.push_back(mo_space_idx);
                        partial_blocks.push_back(new_block);
                    }
                }
            }
        }
        else
        {
            throw std::runtime_error("Index \"" + index + "\" is not defined.");
        }
        final_blocks = partial_blocks;
    }
    return final_blocks;
}

LabeledBlockedTensor::LabeledBlockedTensor(
    BlockedTensor BT, const std::vector<std::string> &indices, double factor)
    : BT_(BT), indices_(indices), factor_(factor)
{
    if (BT_.rank() != indices.size())
        throw std::runtime_error("Labeled tensor does not have correct number "
                                 "of indices for underlying tensor's rank");
}

std::string LabeledBlockedTensor::str() const
{
    std::string s(BT_.name());
    s += "[" + indices::to_string(indices_) + "]";
    return s;
}

void LabeledBlockedTensor::operator=(const LabeledBlockedTensor &rhs)
{
    try
    {
        add(rhs, 1.0, 0.0);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " = " + rhs.str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::operator+=(const LabeledBlockedTensor &rhs)
{
    try
    {
        add(rhs, 1.0, 1.0);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " += " + rhs.str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::operator-=(const LabeledBlockedTensor &rhs)
{
    try
    {
        add(rhs, -1.0, 1.0);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " -= " + rhs.str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::add(const LabeledBlockedTensor &rhs, double alpha,
                               double beta)
{
    std::vector<std::vector<size_t>> rhs_keys = rhs.label_to_block_keys();

    std::vector<size_t> perm =
        indices::permutation_order(indices_, rhs.indices_);

    // Loop over all keys of the rhs
    for (std::vector<size_t> &rhs_key : rhs_keys)
    {
        // Map the rhs key to the lhs key
        std::vector<size_t> lhs_key;
        for (size_t p : perm)
        {
            lhs_key.push_back(rhs_key[p]);
        }

        bool do_add = true;
        // In expert mode if a contraction cannot be performed
        if (BlockedTensor::expert_mode())
        {
            if (not BT().is_block(lhs_key))
                do_add = false;
            if (not rhs.BT().is_block(rhs_key))
                do_add = false;
        }
        if (do_add)
        {
            // Call LabeledTensor's operation
            Tensor LHS = BT().block(lhs_key);
            const Tensor RHS = rhs.BT().block(rhs_key);

            if (LHS == RHS)
                throw std::runtime_error("Self assignment is not allowed.");
            if (LHS.rank() != RHS.rank())
                throw std::runtime_error(
                    "Permuted tensors do not have same rank");
            LHS.permute(RHS, indices_, rhs.indices_, alpha * rhs.factor(),
                        beta);
        }
    }
}

LabeledBlockedTensorProduct LabeledBlockedTensor::
operator*(const LabeledBlockedTensor &rhs)
{
    return LabeledBlockedTensorProduct(*this, rhs);
}

LabeledBlockedTensorAddition LabeledBlockedTensor::
operator+(const LabeledBlockedTensor &rhs)
{
    return LabeledBlockedTensorAddition(*this, rhs);
}

LabeledBlockedTensorAddition LabeledBlockedTensor::
operator-(const LabeledBlockedTensor &rhs)
{
    return LabeledBlockedTensorAddition(*this, -rhs);
}

void LabeledBlockedTensor::operator=(const LabeledBlockedTensorProduct &rhs)
{
    try
    {
        contract(rhs, true, true);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " = " + rhs.str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::operator+=(const LabeledBlockedTensorProduct &rhs)
{
    try
    {
        contract(rhs, false, true);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " += " + rhs.str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::operator-=(const LabeledBlockedTensorProduct &rhs)
{
    try
    {
        contract(rhs, false, false);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " -= " + rhs.str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::operator=(const LabeledBlockedTensorBatchedProduct &rhs)
{
    try
    {
        contract_batched(rhs, true, true);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " = " + rhs.get_contraction().str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::operator+=(const LabeledBlockedTensorBatchedProduct &rhs)
{
    try
    {
        contract_batched(rhs, false, true);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " += " + rhs.get_contraction().str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::operator-=(const LabeledBlockedTensorBatchedProduct &rhs)
{
    try
    {
        contract_batched(rhs, false, false);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " -= " + rhs.get_contraction().str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::contract(const LabeledBlockedTensorProduct &rhs,
                                    bool zero_result, bool add, bool optimize_order)
{
    size_t nterms = rhs.size();

    // Check for self assignment
    for (size_t n = 0; n < nterms; ++n)
    {
        const BlockedTensor &bt = rhs[n].BT();
        if (BT_ == bt)
        {
            throw std::runtime_error(
                "Tensor contractions does not support self assignment.");
        }
    }

    // Find the unique indices in the contraction
    std::vector<std::string> unique_indices;
    for (size_t n = 0; n < nterms; ++n)
    {
        for (const std::string &index : rhs[n].indices())
        {
            unique_indices.push_back(index);
        }
    }
    sort(unique_indices.begin(), unique_indices.end());
    unique_indices.erase(
        std::unique(unique_indices.begin(), unique_indices.end()),
        unique_indices.end());

    std::vector<std::vector<size_t>> unique_indices_keys =
        BlockedTensor::label_to_block_keys(unique_indices);
    std::map<std::string, size_t> index_map;
    {
        size_t k = 0;
        for (const std::string &index : unique_indices)
        {
            index_map[index] = k;
            k++;
        }
    }

    if (zero_result)
    {
        // Zero the results blocks
        for (const std::vector<size_t> &uik : unique_indices_keys)
        {
            std::vector<size_t> result_key;
            for (const std::string &index : indices())
            {
                result_key.push_back(uik[index_map[index]]);
            }
            if (BlockedTensor::expert_mode_)
            {
                if (BT_.is_block(result_key))
                {
                    BT_.block(result_key).zero();
                }
            }
            else
            {
                BT_.block(result_key).zero();
            }
        }
    }

    // Setup and perform contractions
    for (const std::vector<size_t> &uik : unique_indices_keys)
    {
        std::vector<size_t> result_key;
        for (const std::string &index : indices())
        {
            result_key.push_back(uik[index_map[index]]);
        }

        bool do_contract = true;
        // In expert mode if a contraction cannot be performed
        if (BlockedTensor::expert_mode_)
        {
            if (not BT().is_block(result_key))
                do_contract = false;
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = rhs[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                if (not lbt.BT().is_block(term_key))
                    do_contract = false;
            }
        }

        if (do_contract)
        {
            LabeledTensor result(BT().block(result_key), indices(), factor());

            LabeledTensorContraction prod;
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = rhs[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                const LabeledTensor term(lbt.BT().block(term_key),
                                         lbt.indices(), lbt.factor());
                prod *= term;
            }
            if (add)
            {
//                result += prod;
                result.contract(prod, false, true, optimize_order);
            }
            else
            {
//                result -= prod;
                result.contract(prod, false, false, optimize_order);
            }
        }
    }
}

void LabeledBlockedTensor::set(const LabeledBlockedTensor &to)
{
    BT_ = to.BT_;
    indices_ = to.indices_;
    factor_ = to.factor_;
}

void LabeledBlockedTensor::contract_by_tensor(const LabeledBlockedTensorProduct &rhs,
                                    bool zero_result, bool add, bool optimize_order)
{
    size_t nterms = rhs.size();
    // Check for self assignment
    for (size_t n = 0; n < nterms; ++n)
    {
        const BlockedTensor &bt = rhs[n].BT();
        if (BT_ == bt)
        {
            throw std::runtime_error(
                "Tensor contractions does not support self assignment.");
        }
    }

    std::vector<std::vector<size_t>> unique_indices_keys;
    std::map<std::string, size_t> index_map;
    // In expert mode if a contraction cannot be performed
    if (BlockedTensor::expert_mode_)
    {
        // Find the unique indices in the contraction
        std::vector<std::string> unique_indices;
        for (size_t n = 0; n < nterms; ++n)
        {
            for (const std::string &index : rhs[n].indices())
            {
                unique_indices.push_back(index);
            }
        }
        sort(unique_indices.begin(), unique_indices.end());
        unique_indices.erase(
            std::unique(unique_indices.begin(), unique_indices.end()),
            unique_indices.end());

        unique_indices_keys = BlockedTensor::label_to_block_keys(unique_indices);
        {
            size_t k = 0;
            for (const std::string &index : unique_indices)
            {
                index_map[index] = k;
                k++;
            }
        }

        std::vector<std::vector<size_t>> unique_indices_keys_expert =
            unique_indices_keys;
        unique_indices_keys.clear();
        for (const std::vector<size_t> &uik : unique_indices_keys_expert)
        {
            std::vector<size_t> result_key;
            for (const std::string &index : indices())
            {
                result_key.push_back(uik[index_map[index]]);
            }
            if (not BT().is_block(result_key))
                continue;
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = rhs[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                if (not lbt.BT().is_block(term_key))
                    continue;
            }
            unique_indices_keys.push_back(uik);
        }
    }

    std::vector<size_t> perm(nterms);
    std::vector<size_t> best_perm(nterms);
    std::iota(perm.begin(), perm.end(), 0);
    std::pair<double, double> best_cpu_memory_cost(1.0e200, 1.0e200);

    if (optimize_order) {
        throw std::runtime_error(
            "Order optimization in LabeledBlockedTensor::contract_by_tensor not implemented.");
//        do
//        {
//            std::pair<double, double> cpu_memory_cost =
//                rhs.compute_contraction_cost(perm);
//            if (cpu_memory_cost.first < best_cpu_memory_cost.first)
//            {
//                best_perm = perm;
//                best_cpu_memory_cost = cpu_memory_cost;
//            }
//        } while (std::next_permutation(perm.begin(), perm.end()));
        // at this point 'best_perm' should be used to perform contraction in
        // optimal order.
    } else {
        best_perm = perm;
    }

    const LabeledBlockedTensor &Aref = rhs[best_perm[0]];
    LabeledBlockedTensor A(Aref.BT_, Aref.indices_, Aref.factor_);
    int maxn = int(nterms) - 2;
    for (int n = 0; n < maxn; ++n)
    {
        const LabeledBlockedTensor &B = rhs[best_perm[n + 1]];

        std::vector<Indices> AB_indices =
            indices::determine_contraction_result_from_indices(A.indices(), B.indices());
        const Indices &AB_common_idx = AB_indices[0];
        const Indices &A_fix_idx = AB_indices[1];
        const Indices &B_fix_idx = AB_indices[2];
        Indices indices;

        for (size_t i = 0; i < AB_common_idx.size(); ++i)
        {
            // If a common index is also found in the rhs it's a Hadamard index
            if (std::find(this->indices().begin(), this->indices().end(),
                          AB_common_idx[i]) != this->indices().end())
            {
                indices.push_back(AB_common_idx[i]);
            }
        }

        for (size_t i = 0; i < A_fix_idx.size(); ++i)
        {
            indices.push_back(A_fix_idx[i]);
        }
        for (size_t i = 0; i < B_fix_idx.size(); ++i)
        {
            indices.push_back(B_fix_idx[i]);
        }

        std::vector<std::vector<size_t>> AB_block_keys;
        if (BlockedTensor::expert_mode_)
        {
            for (const std::vector<size_t> &uik : unique_indices_keys)
            {
                std::vector<size_t> term_key;
                for (const std::string &index : indices)
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                AB_block_keys.push_back(term_key);
            }
        } else {
            AB_block_keys = BlockedTensor::label_to_block_keys(indices);
        }

        std::vector<std::string> AB_blocks;
        for (const std::vector<size_t>& block_key : AB_block_keys)
        {
            std::string block = "";
            for (size_t i = 0, maxI = block_key.size() - 1; i < maxI; ++i) {
                block += BlockedTensor::mo_space(block_key[i]).name() + ",";
            }
            block += BlockedTensor::mo_space(block_key[block_key.size() - 1]).name();
            AB_blocks.push_back(block);
        }

        BlockedTensor btAB = BlockedTensor::build(CoreTensor, A.BT().name() + " * " + B.BT().name(), AB_blocks);

        LabeledBlockedTensor AB(btAB, indices);

        AB.contract(A * B, true, true, false);

        A.set(AB);
    }
    const LabeledBlockedTensor &B = rhs[best_perm[nterms - 1]];

    this->contract(A * B, zero_result, add, false);
}

void LabeledBlockedTensor::contract_batched(const LabeledBlockedTensorBatchedProduct &rhs_batched,
                                    bool zero_result, bool add, bool optimize_order)
{
    const LabeledBlockedTensorProduct &rhs = rhs_batched.get_contraction();
    const Indices &batched_indices = rhs_batched.get_batched_indices();

    size_t nterms = rhs.size();

    // Check for self assignment
    for (size_t n = 0; n < nterms; ++n)
    {
        const BlockedTensor &bt = rhs[n].BT();
        if (BT_ == bt)
        {
            throw std::runtime_error(
                "Tensor contractions does not support self assignment.");
        }
    }

    // Find the unique indices in the contraction
    std::vector<std::string> unique_indices;
    for (size_t n = 0; n < nterms; ++n)
    {
        for (const std::string &index : rhs[n].indices())
        {
            unique_indices.push_back(index);
        }
    }
    sort(unique_indices.begin(), unique_indices.end());
    unique_indices.erase(
        std::unique(unique_indices.begin(), unique_indices.end()),
        unique_indices.end());

    std::vector<std::vector<size_t>> unique_indices_keys =
        BlockedTensor::label_to_block_keys(unique_indices);
    std::map<std::string, size_t> index_map;
    {
        size_t k = 0;
        for (const std::string &index : unique_indices)
        {
            index_map[index] = k;
            k++;
        }
    }

    if (zero_result)
    {
        // Zero the results blocks
        for (const std::vector<size_t> &uik : unique_indices_keys)
        {
            std::vector<size_t> result_key;
            for (const std::string &index : indices())
            {
                result_key.push_back(uik[index_map[index]]);
            }
            if (BlockedTensor::expert_mode_)
            {
                if (BT_.is_block(result_key))
                {
                    BT_.block(result_key).zero();
                }
            }
            else
            {
                BT_.block(result_key).zero();
            }
        }
    }

    // Setup and perform contractions
    for (const std::vector<size_t> &uik : unique_indices_keys)
    {
        std::vector<size_t> result_key;
        for (const std::string &index : indices())
        {
            result_key.push_back(uik[index_map[index]]);
        }

        bool do_contract = true;
        // In expert mode if a contraction cannot be performed
        if (BlockedTensor::expert_mode_)
        {
            if (not BT().is_block(result_key))
                do_contract = false;
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = rhs[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                if (not lbt.BT().is_block(term_key))
                    do_contract = false;
            }
        }

        if (do_contract)
        {
            LabeledTensor result(BT().block(result_key), indices(), factor());

            LabeledTensorContraction prod;
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = rhs[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                const LabeledTensor term(lbt.BT().block(term_key),
                                         lbt.indices(), lbt.factor());
                prod *= term;
            }
            LabeledTensorBatchedContraction prod_batched(prod, batched_indices);
            if (add)
            {
//                result += prod_batched;
                result.contract_batched(prod_batched, false, true, optimize_order);
            }
            else
            {
//                result -= prod_batched;
                result.contract_batched(prod_batched, false, false, optimize_order);
            }
        }
    }
}

void LabeledBlockedTensor::contract_batched_by_tensor(const LabeledBlockedTensorBatchedProduct &rhs_batched,
                                    bool zero_result, bool add, bool optimize_order)
{
    const LabeledBlockedTensorProduct &rhs = rhs_batched.get_contraction();

    size_t nterms = rhs.size();
    // Check for self assignment
    for (size_t n = 0; n < nterms; ++n)
    {
        const BlockedTensor &bt = rhs[n].BT();
        if (BT_ == bt)
        {
            throw std::runtime_error(
                "Tensor contractions does not support self assignment.");
        }
    }

    std::vector<std::vector<size_t>> unique_indices_keys;
    std::map<std::string, size_t> index_map;
    // In expert mode if a contraction cannot be performed
    if (BlockedTensor::expert_mode_)
    {
        // Find the unique indices in the contraction
        std::vector<std::string> unique_indices;
        for (size_t n = 0; n < nterms; ++n)
        {
            for (const std::string &index : rhs[n].indices())
            {
                unique_indices.push_back(index);
            }
        }
        sort(unique_indices.begin(), unique_indices.end());
        unique_indices.erase(
            std::unique(unique_indices.begin(), unique_indices.end()),
            unique_indices.end());

        unique_indices_keys = BlockedTensor::label_to_block_keys(unique_indices);
        {
            size_t k = 0;
            for (const std::string &index : unique_indices)
            {
                index_map[index] = k;
                k++;
            }
        }

        std::vector<std::vector<size_t>> unique_indices_keys_expert =
            unique_indices_keys;
        unique_indices_keys.clear();
        for (const std::vector<size_t> &uik : unique_indices_keys_expert)
        {
            std::vector<size_t> result_key;
            for (const std::string &index : indices())
            {
                result_key.push_back(uik[index_map[index]]);
            }
            if (not BT().is_block(result_key))
                continue;
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = rhs[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                if (not lbt.BT().is_block(term_key))
                    continue;
            }
            unique_indices_keys.push_back(uik);
        }
    }

    std::vector<size_t> perm(nterms);
    std::vector<size_t> best_perm(nterms);
    std::iota(perm.begin(), perm.end(), 0);
    std::pair<double, double> best_cpu_memory_cost(1.0e200, 1.0e200);

    if (optimize_order) {
        throw std::runtime_error(
            "Order optimization in LabeledBlockedTensor::contract_by_tensor not implemented.");
//        do
//        {
//            std::pair<double, double> cpu_memory_cost =
//                rhs.compute_contraction_cost(perm);
//            if (cpu_memory_cost.first < best_cpu_memory_cost.first)
//            {
//                best_perm = perm;
//                best_cpu_memory_cost = cpu_memory_cost;
//            }
//        } while (std::next_permutation(perm.begin(), perm.end()));
        // at this point 'best_perm' should be used to perform contraction in
        // optimal order.
    } else {
        best_perm = perm;
    }

    // Find the indices to be batched in result labeled tensor.
    const Indices &batched_indices = rhs_batched.get_batched_indices();
    size_t batched_size = batched_indices.size();
    std::vector<size_t> slicing_axis(batched_size);
    for (size_t l = 0; l < batched_size; ++l) {
        auto it = std::find(indices_.begin(), indices_.end(), batched_indices[l]);
        if (it != indices_.end()) {
            slicing_axis[l] = std::distance(indices_.begin(), it);
        } else {
            throw std::runtime_error("Slicing indices do not exist in tensor indices.");
        }
    }

    // Determine if the result need permutation to permute the batched indices to the front
    bool permute_flag = false;
    for (size_t l = 0; l < batched_size; ++l) {
        if (slicing_axis[l] != l) {
            permute_flag = true;
            break;
        }
    }

    // Determine batched dimensions.
    Indices permuted_indices;

    // Permute result labeled tensor.
    LabeledBlockedTensor Llbtp(BT(), indices());
    if (permute_flag) {
        permuted_indices.insert(permuted_indices.end(), batched_indices.begin(), batched_indices.end());
        for (size_t l = 0, l_max = numdim(); l < l_max; ++l) {
            if (std::find(slicing_axis.begin(), slicing_axis.end(),l) == slicing_axis.end()) {
                permuted_indices.push_back(indices_[l]);
            }
        }
        // Determine blocks
        std::vector<size_t> perm =
            indices::permutation_order(permuted_indices, indices());
        std::vector<std::string> L_blocks;
        for (const std::string & block : BT().block_labels_) {
            std::vector<std::string> block_vec = indices::split(block);
            std::vector<std::string> L_block_vec(block_vec.size());
            for (size_t i = 0; i < block_vec.size(); ++i) {
                L_block_vec[i] = block_vec[perm[i]];
            }
            L_blocks.push_back(indices::to_string(L_block_vec, ","));
        }

        BlockedTensor Lbtp = BlockedTensor::build(CoreTensor, BT().name() + " permute", L_blocks);
        LabeledBlockedTensor Ltemp(Lbtp, permuted_indices);
        Ltemp = Llbtp;
        Llbtp.set(Ltemp);
    }

    throw std::runtime_error(
        "LabeledBlockedTensor::contract_batched_by_tensor implementation unfinished.");
}

void LabeledBlockedTensor::operator=(const LabeledBlockedTensorAddition &rhs)
{
    BT_.zero();
    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind)
    {
        const LabeledBlockedTensor &labeledTensor = rhs[ind];
        add(labeledTensor, 1.0, 1.0);
    }
}

void LabeledBlockedTensor::operator+=(const LabeledBlockedTensorAddition &rhs)
{
    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind)
    {
        const LabeledBlockedTensor &labeledTensor = rhs[ind];
        add(labeledTensor, 1.0, 1.0);
    }
}

void LabeledBlockedTensor::operator-=(const LabeledBlockedTensorAddition &rhs)
{
    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind)
    {
        const LabeledBlockedTensor &labeledTensor = rhs[ind];
        add(labeledTensor, -1.0, 1.0);
    }
}

void LabeledBlockedTensor::operator*=(double scale)
{
    std::vector<std::vector<size_t>> keys = label_to_block_keys();

    // Loop over all keys and scale blocks
    for (std::vector<size_t> &key : keys)
    {
        BT_.block(key).scale(scale);
    }
}

void LabeledBlockedTensor::operator/=(double scale)
{
    std::vector<std::vector<size_t>> keys = label_to_block_keys();

    // Loop over all keys and scale blocks
    for (std::vector<size_t> &key : keys)
    {
        BT_.block(key).scale(1.0 / scale);
    }
}

LabeledBlockedTensorDistributive LabeledBlockedTensor::
operator*(const LabeledBlockedTensorAddition &rhs)
{
    return LabeledBlockedTensorDistributive(*this, rhs);
}

void LabeledBlockedTensor::
operator=(const LabeledBlockedTensorDistributive &rhs)
{
    std::vector<std::vector<size_t>> lhs_keys = label_to_block_keys();

    // Loop over all keys of the rhs
    for (std::vector<size_t> &lhs_key : lhs_keys)
    {
        BT_.block(lhs_key).zero();
    }

    for (const LabeledBlockedTensor &B : rhs.B())
    {
        *this += const_cast<LabeledBlockedTensor &>(rhs.A()) *
                 const_cast<LabeledBlockedTensor &>(B);
    }
}

void LabeledBlockedTensor::
operator+=(const LabeledBlockedTensorDistributive &rhs)
{
    for (const LabeledBlockedTensor &B : rhs.B())
    {
        *this += const_cast<LabeledBlockedTensor &>(rhs.A()) *
                 const_cast<LabeledBlockedTensor &>(B);
    }
}

void LabeledBlockedTensor::
operator-=(const LabeledBlockedTensorDistributive &rhs)
{
    for (const LabeledBlockedTensor &B : rhs.B())
    {
        *this -= const_cast<LabeledBlockedTensor &>(rhs.A()) *
                 const_cast<LabeledBlockedTensor &>(B);
    }
}

LabeledBlockedTensorDistributive LabeledBlockedTensorAddition::
operator*(const LabeledBlockedTensor &other)
{
    return LabeledBlockedTensorDistributive(other, *this);
}

LabeledBlockedTensorAddition &LabeledBlockedTensorAddition::
operator*(double scalar)
{
    // distribute the scalar to each term
    for (LabeledBlockedTensor &T : tensors_)
    {
        T *= scalar;
    }

    return *this;
}

LabeledBlockedTensorAddition &LabeledBlockedTensorAddition::operator-()
{
    for (LabeledBlockedTensor &T : tensors_)
    {
        T *= -1.0;
    }

    return *this;
}

std::string LabeledBlockedTensorProduct::str() const
{
    std::vector<std::string> vec_str;
    for (const auto &tensor : tensors_)
    {
        vec_str.push_back(tensor.str());
    }
    return indices::to_string(vec_str, " * ");
}

LabeledBlockedTensorProduct::operator double() const
{
    double result = 0.0;

    size_t nterms = this->size();

    // Find the unique indices in the contraction
    std::vector<std::string> unique_indices;
    for (size_t n = 0; n < nterms; ++n)
    {
        for (const std::string &index : tensors_[n].indices())
        {
            unique_indices.push_back(index);
        }
    }
    sort(unique_indices.begin(), unique_indices.end());
    unique_indices.erase(
        std::unique(unique_indices.begin(), unique_indices.end()),
        unique_indices.end());

    std::vector<std::vector<size_t>> unique_indices_keys =
        BlockedTensor::label_to_block_keys(unique_indices);
    std::map<std::string, size_t> index_map;
    {
        size_t k = 0;
        for (const std::string &index : unique_indices)
        {
            index_map[index] = k;
            k++;
        }
    }

    // Setup and perform contractions
    for (const std::vector<size_t> &uik : unique_indices_keys)
    {

        bool do_contract = true;
        // In expert mode if a contraction cannot be performed
        if (BlockedTensor::expert_mode())
        {
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = tensors_[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                if (not lbt.BT().is_block(term_key))
                    do_contract = false;
            }
        }

        if (do_contract)
        {
            LabeledTensorContraction prod;
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = tensors_[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                const LabeledTensor term(lbt.BT().block(term_key),
                                         lbt.indices(), lbt.factor());
                prod *= term;
            }
            result += prod;
        }
    }

    return result;
}

std::vector<std::string> spin_cases(const std::vector<std::string> &in_str_vec)
{
    std::vector<std::string> out_str_vec;
    for (const std::string &s : in_str_vec)
    {
        if (s.size() % 2 == 1)
        {
            throw std::runtime_error("String \"" + s +
                                     "\" passed to spin_cases() is not valid.");
        }
        size_t n = s.size() / 2;
        for (size_t i = 0; i < n + 1; ++i)
        {
            std::string mod_s = s;
            std::transform(mod_s.begin(), mod_s.end(), mod_s.begin(),
                           ::tolower);
            for (size_t j = n - i; j < n; ++j)
            {
                mod_s[j] = ::toupper(mod_s[j]);
                mod_s[n + j] = ::toupper(mod_s[n + j]);
            }
            out_str_vec.push_back(mod_s);
        }
    }
    return out_str_vec;
}

LabeledBlockedTensorBatchedProduct batched(const string &batched_indices, const LabeledBlockedTensorProduct &product)
{
    return LabeledBlockedTensorBatchedProduct(product, indices::split(batched_indices));
}

}
