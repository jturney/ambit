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
 * You should have received a copy of the GNU Lesser General Public License
 * along with ambit; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

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

std::vector<std::string> BlockedTensor::block_labels() const
{
    std::vector<std::string> labels;
    for (const auto &this_block_tensor : blocks_)
    {
        // Grab the orbital spaces names
        std::string block_label;
        for (size_t ms : this_block_tensor.first)
        {
            block_label += mo_spaces_[ms].name();
        }
        labels.push_back(block_label);
    }
    return labels;
}

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

std::vector<std::string> BlockedTensor::indices_to_block_labels(
    const Indices &indices,
    const std::vector<std::vector<size_t>> &unique_indices_keys,
    const std::map<std::string, size_t> &index_map, bool full_contraction)
{
    std::vector<std::string> blocks;
    if (full_contraction)
    {
        std::vector<std::vector<size_t>> block_keys =
            BlockedTensor::label_to_block_keys(indices);
        for (const std::vector<size_t> &block_key : block_keys)
        {
            std::string block_label;
            for (size_t ms : block_key)
            {
                block_label += BlockedTensor::mo_space(ms).name();
            }
            blocks.push_back(block_label);
        }
    }
    else
    {
        size_t max_path = 1;
        for (const std::string &index : indices)
        {
            max_path *= BlockedTensor::index_to_mo_spaces_[index].size();
        }
        std::set<std::vector<size_t>> block_set;
        for (const std::vector<size_t> &uik : unique_indices_keys)
        {
            std::vector<size_t> term_key;
            for (const std::string &index : indices)
            {
                term_key.push_back(uik[index_map.at(index)]);
            }
            block_set.insert(term_key);
            if (block_set.size() == max_path)
                break;
        }
        for (const std::vector<size_t> &block_key : block_set)
        {
            std::string block_label;
            for (size_t ms : block_key)
            {
                block_label += BlockedTensor::mo_space(ms).name();
            }
            blocks.push_back(block_label);
        }
    }
    return blocks;
}

std::vector<std::string> BlockedTensor::reduce_rank_block_labels(
    const Indices &indices, const Indices &full_rank_indices,
    const std::map<std::vector<size_t>, Tensor> &blocks, bool full_contraction)
{
    std::map<std::string, size_t> sub_index_map;
    std::vector<std::vector<size_t>> sub_uiks;
    sub_uiks.reserve(blocks.size());
    if (not full_contraction)
    {
        std::vector<std::vector<size_t>> block_keys =
            BlockedTensor::label_to_block_keys(full_rank_indices);
        for (const auto &block : block_keys)
        {
            if (blocks.count(block) != 0)
            {
                sub_uiks.push_back(block);
            }
        }
        size_t count = 0;
        for (const std::string &index : full_rank_indices)
        {
            sub_index_map[index] = count++;
        }
    }
    return BlockedTensor::indices_to_block_labels(
        indices, sub_uiks, sub_index_map, full_contraction);
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

void BlockedTensor::set_block(const std::string &indices, Tensor t)
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
    return set_block(key, t);
}

void BlockedTensor::set_block(const std::vector<size_t> &key, Tensor t)
{
    // check that the new tensor has a size compatible with the block
    for (size_t s = 0, max_s = key.size(); s < max_s; s++)
    {
        if (t.dims()[s] != BlockedTensor::mo_space(key[s]).dim())
        {
            throw std::runtime_error("BlockedTensor::set_block the size of the "
                                     "tensor is not consistent with the block");
        }
    }
    blocks_[key] = t;
    rank_ = key.size();
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
            [&](const std::vector<size_t> &indices, double &value) {
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
            [&](const std::vector<size_t> &indices, const double &value) {
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

void save(BlockedTensor bt, const std::string &filename, bool overwrite)
{
    // check if file exists or not
    struct stat buf;
    if (stat(filename.c_str(), &buf) == 0)
    {
        if (overwrite)
        {
            // delete the file
            if (remove(filename.c_str()) != 0)
            {
                std::string msg = "Error when deleting " + filename;
                perror(msg.c_str());
            }
        }
        else
        {
            std::string error = "File " + filename + " already exists.";
            throw std::runtime_error(error);
        }
    }
    // create the file
    std::ofstream out(filename.c_str(), std::ios_base::binary);

    auto block_labels = bt.block_labels();

    // write the name
    auto name = bt.name();
    size_t size = name.size();
    out.write(reinterpret_cast<char *>(&size), sizeof(size_t));
    out.write(&name[0], size);

    // write the number of blocks
    size_t num_blocks = block_labels.size();
    out.write(reinterpret_cast<char *>(&num_blocks), sizeof(size_t));

    // write the block labels
    for (const std::string &block : block_labels)
    {
        size_t block_label_size = block.size();
        out.write(reinterpret_cast<char *>(&block_label_size), sizeof(size_t));
        out.write(&block[0], block_label_size);
    }

    // write the blocks
    for (const std::string &block : block_labels)
    {
        auto t = bt.block(block);
        write_tensor_to_file(t, out);
    }
}

void load(BlockedTensor &bt, const std::string &filename)
{
    // check if file exists or not
    std::ifstream in(filename.c_str(), std::ios_base::binary);
    if (!in.good())
    {
        std::string error = "File " + filename + " does not exist.";
        throw std::runtime_error(error);
    }

    // read the name
    std::string name;
    size_t size = 0;
    in.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    name.resize(size);
    in.read(&name[0], size);

    // read the number of blocks
    size_t num_blocks = 0;
    in.read(reinterpret_cast<char *>(&num_blocks), sizeof(size_t));

    // read the block labels
    std::vector<std::string> block_labels;
    for (size_t b = 0; b < num_blocks; b++)
    {
        std::string block;
        size_t block_label_size = 0;
        in.read(reinterpret_cast<char *>(&block_label_size), sizeof(size_t));
        block.resize(block_label_size);
        in.read(&block[0], block_label_size);
        block_labels.push_back(block);
    }

    // read tensor from file
    for (const std::string &block : block_labels)
    {
        Tensor t;
        read_tensor_from_file(t, in);
        bt.set_block(block, t);
    }

    // 4. close the file
    in.close();
}

BlockedTensor load_blocked_tensor(const std::string &filename)
{
    BlockedTensor bt;
    load(bt, filename);
    return bt;
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

void LabeledBlockedTensor::
operator=(const LabeledBlockedTensorBatchedProduct &rhs)
{
    try
    {
        contract_batched(rhs, true, true);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() + " = " +
                          rhs.get_contraction().str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::
operator+=(const LabeledBlockedTensorBatchedProduct &rhs)
{
    try
    {
        contract_batched(rhs, false, true);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() +
                          " += " + rhs.get_contraction().str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::
operator-=(const LabeledBlockedTensorBatchedProduct &rhs)
{
    try
    {
        contract_batched(rhs, false, false);
    }
    catch (std::exception &e)
    {
        std::string msg = "\n" + this->str() +
                          " -= " + rhs.get_contraction().str() + " <- " +
                          std::string(e.what());
        throw std::runtime_error(msg);
    }
}

void LabeledBlockedTensor::contract_pair(
    const LabeledBlockedTensorProduct &rhs, bool zero_result, bool add,
    std::shared_ptr<std::tuple<std::vector<std::vector<size_t>>,
                               std::map<std::string, size_t>>> &block_info_ptr)
{
    size_t nterms = rhs.size();

    // Find the unique indices in the contraction
    if (!block_info_ptr)
    {
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
        block_info_ptr =
            std::make_shared<std::tuple<std::vector<std::vector<size_t>>,
                                        std::map<std::string, size_t>>>(
                std::make_tuple(unique_indices_keys, index_map));
    }
    std::vector<std::vector<size_t>> &unique_indices_keys =
        std::get<0>(*block_info_ptr);
    std::map<std::string, size_t> &index_map = std::get<1>(*block_info_ptr);

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

            result.contract(prod, false, add, false);
        }
    }
}

void LabeledBlockedTensor::set(const LabeledBlockedTensor &to)
{
    BT_ = to.BT_;
    indices_ = to.indices_;
    factor_ = to.factor_;
}

void LabeledBlockedTensor::contract(const LabeledBlockedTensorProduct &rhs,
                                    bool zero_result, bool add,
                                    bool optimize_order)
{
    std::vector<std::shared_ptr<BlockedTensor>> inter_AB_tensors(rhs.size() -
                                                                 2);
    std::shared_ptr<std::tuple<bool, std::vector<std::vector<size_t>>,
                               std::map<std::string, size_t>>>
        expert_info_ptr;
    std::vector<std::shared_ptr<std::tuple<std::vector<std::vector<size_t>>,
                                           std::map<std::string, size_t>>>>
        inter_block_info_ptrs(rhs.size() - 1);
    contract(rhs, zero_result, add, optimize_order, inter_AB_tensors,
             expert_info_ptr, inter_block_info_ptrs);
}

void LabeledBlockedTensor::contract(
    const LabeledBlockedTensorProduct &rhs, bool zero_result, bool add,
    bool optimize_order,
    std::vector<std::shared_ptr<BlockedTensor>> &inter_AB_tensors,
    std::shared_ptr<std::tuple<bool, std::vector<std::vector<size_t>>,
                               std::map<std::string, size_t>>> &expert_info_ptr,
    std::vector<std::shared_ptr<std::tuple<std::vector<std::vector<size_t>>,
                                           std::map<std::string, size_t>>>>
        &inter_block_info_ptrs)
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

    // In expert mode if a contraction cannot be performed
    if (!expert_info_ptr)
    {
        std::vector<std::vector<size_t>> unique_indices_keys;
        std::map<std::string, size_t> index_map;
        bool full_contraction = true;
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

            unique_indices_keys =
                BlockedTensor::label_to_block_keys(unique_indices);
            size_t full_contraction_size = unique_indices_keys.size();
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
                else if (zero_result)
                {
                    BT_.block(result_key).zero();
                }
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
            if (unique_indices_keys.size() == 0)
            {
                return;
            }
            if (full_contraction_size > unique_indices_keys.size())
            {
                full_contraction = false;
            }
        }
        expert_info_ptr =
            std::make_shared<std::tuple<bool, std::vector<std::vector<size_t>>,
                                        std::map<std::string, size_t>>>(
                std::make_tuple(full_contraction, unique_indices_keys,
                                index_map));
    }
    bool full_contraction = std::get<0>(*expert_info_ptr);
    std::vector<std::vector<size_t>> &unique_indices_keys =
        std::get<1>(*expert_info_ptr);
    std::map<std::string, size_t> &index_map = std::get<2>(*expert_info_ptr);

    std::vector<size_t> perm(nterms);
    std::vector<size_t> best_perm(nterms);
    std::iota(perm.begin(), perm.end(), 0);
    std::pair<double, double> best_cpu_memory_cost(1.0e200, 1.0e200);

    if (optimize_order)
    {
        do
        {
            std::pair<double, double> cpu_memory_cost =
                rhs.compute_contraction_cost(perm, unique_indices_keys,
                                             index_map, full_contraction);
            if (cpu_memory_cost.first < best_cpu_memory_cost.first)
            {
                best_perm = perm;
                best_cpu_memory_cost = cpu_memory_cost;
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
        // at this point 'best_perm' should be used to perform contraction in
        // optimal order.
    }
    else
    {
        best_perm = perm;
    }

    const LabeledBlockedTensor &Aref = rhs[best_perm[0]];
    LabeledBlockedTensor A(Aref.BT_, Aref.indices_, Aref.factor_);
    int maxn = int(nterms) - 2;
    for (int n = 0; n < maxn; ++n)
    {
        const LabeledBlockedTensor &B = rhs[best_perm[n + 1]];

        std::vector<Indices> AB_indices =
            indices::determine_contraction_result_from_indices(A.indices(),
                                                               B.indices());
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

        std::vector<std::string> AB_blocks =
            BlockedTensor::indices_to_block_labels(indices, unique_indices_keys,
                                                   index_map, full_contraction);

        if (!inter_AB_tensors[n])
        {
            inter_AB_tensors[n] =
                std::make_shared<BlockedTensor>(BlockedTensor::build(
                    CoreTensor, A.BT().name() + " * " + B.BT().name(),
                    AB_blocks));
        }
        LabeledBlockedTensor AB(*(inter_AB_tensors[n]), indices);

        AB.contract_pair(A * B, true, true, inter_block_info_ptrs[n]);

        A.set(AB);
    }
    const LabeledBlockedTensor &B = rhs[best_perm[nterms - 1]];

    contract_pair(A * B, zero_result, add, inter_block_info_ptrs[maxn]);
}

void LabeledBlockedTensor::contract_batched(
    const LabeledBlockedTensorBatchedProduct &rhs_batched, bool zero_result,
    bool add, bool optimize_order)
{
    const LabeledBlockedTensorProduct &rhs = rhs_batched.get_contraction();
    const Indices &batched_indices = rhs_batched.get_batched_indices();
    size_t batched_size = batched_indices.size();

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
    bool full_contraction = true;
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
        Indices sorted_batched_indices(batched_indices);
        sort(sorted_batched_indices.begin(), sorted_batched_indices.end());
        Indices unbatched_indices;
        std::set_difference(unique_indices.begin(), unique_indices.end(),
                            sorted_batched_indices.begin(),
                            sorted_batched_indices.end(),
                            std::back_inserter(unbatched_indices));
        unique_indices = batched_indices;
        unique_indices.insert(unique_indices.end(), unbatched_indices.begin(),
                              unbatched_indices.end());

        unique_indices_keys =
            BlockedTensor::label_to_block_keys(unique_indices);
        size_t full_contraction_size = unique_indices_keys.size();
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
            else if (zero_result)
            {
                BT_.block(result_key).zero();
            }
            bool do_contraction = true;
            for (size_t n = 0; n < nterms; ++n)
            {
                const LabeledBlockedTensor &lbt = rhs[n];
                std::vector<size_t> term_key;
                for (const std::string &index : lbt.indices())
                {
                    term_key.push_back(uik[index_map[index]]);
                }
                if (not lbt.BT().is_block(term_key))
                {
                    do_contraction = false;
                    break;
                }
            }
            if (do_contraction)
            {
                unique_indices_keys.push_back(uik);
            }
        }
        if (unique_indices_keys.size() == 0)
        {
            return;
        }
        if (full_contraction_size > unique_indices_keys.size())
        {
            full_contraction = false;
        }
    }

    std::vector<size_t> perm(nterms);
    std::vector<size_t> best_perm(nterms);
    std::iota(perm.begin(), perm.end(), 0);
    std::pair<double, double> best_cpu_memory_cost(1.0e200, 1.0e200);

    if (optimize_order)
    {
        do
        {
            std::pair<double, double> cpu_memory_cost =
                rhs.compute_contraction_cost(perm, unique_indices_keys,
                                             index_map, full_contraction);
            if (cpu_memory_cost.first < best_cpu_memory_cost.first)
            {
                best_perm = perm;
                best_cpu_memory_cost = cpu_memory_cost;
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
        // at this point 'best_perm' should be used to perform contraction in
        // optimal order.
    }
    else
    {
        best_perm = perm;
    }

    // Find the indices to be batched in result labeled tensor.
    std::vector<size_t> slicing_axis(batched_size);
    for (size_t l = 0; l < batched_size; ++l)
    {
        auto it =
            std::find(indices_.begin(), indices_.end(), batched_indices[l]);
        if (it != indices_.end())
        {
            slicing_axis[l] = std::distance(indices_.begin(), it);
        }
        else
        {
            throw std::runtime_error(
                "Slicing indices do not exist in tensor indices.");
        }
    }

    // Determine if the result need permutation to permute the batched indices
    // to the front
    bool permute_flag = false;
    for (size_t l = 0; l < batched_size; ++l)
    {
        if (slicing_axis[l] != l)
        {
            permute_flag = true;
            break;
        }
    }

    // Determine batched dimensions.
    Indices permuted_indices;

    // Permute result labeled tensor.
    LabeledBlockedTensor Lt(BT(), indices());
    if (permute_flag)
    {
        permuted_indices = batched_indices;
        for (size_t l = 0, l_max = numdim(); l < l_max; ++l)
        {
            if (std::find(slicing_axis.begin(), slicing_axis.end(), l) ==
                slicing_axis.end())
            {
                permuted_indices.push_back(indices_[l]);
            }
        }
        // Determine blocks
        std::vector<std::string> L_blocks =
            BlockedTensor::reduce_rank_block_labels(
                permuted_indices, indices(), BT().blocks_, full_contraction);

        BlockedTensor Lbtp = BlockedTensor::build(
            CoreTensor, BT().name() + " permute", L_blocks);
        LabeledBlockedTensor Ltemp(Lbtp, permuted_indices);
        Ltemp = Lt;
        Lt.set(Ltemp);
    }

    std::vector<std::vector<bool>> need_slicing(
        nterms, std::vector<bool>(batched_size + 1));

    // Permute tensor indices if the corresponding tensor needs to be batched.
    LabeledBlockedTensorProduct rhsp;
    for (size_t i = 0; i < nterms; ++i)
    {
        const LabeledBlockedTensor &A = rhs[best_perm[i]];
        const Indices &A_indices = A.indices();
        Indices gemm_indices;
        for (const string &s : A_indices)
        {
            auto it =
                std::find(batched_indices.begin(), batched_indices.end(), s);
            if (it != batched_indices.end())
            {
                need_slicing[i][std::distance(batched_indices.begin(), it)] =
                    true;
            }
            else
            {
                gemm_indices.push_back(s);
            }
        }
        Indices permuted_indices;
        for (size_t l = 0; l < batched_size; ++l)
        {
            if (need_slicing[i][l])
            {
                permuted_indices.push_back(batched_indices[l]);
                need_slicing[i][batched_size] = true;
            }
        }
        if (permuted_indices.size() == 0)
        {
            rhsp.operator*(A);
        }
        else
        {
            permuted_indices.insert(permuted_indices.end(),
                                    gemm_indices.begin(), gemm_indices.end());
            if (permuted_indices == A_indices)
            {
                rhsp.operator*(A);
            }
            else
            {
                // Determine blocks
                std::vector<std::string> A_blocks =
                    BlockedTensor::reduce_rank_block_labels(
                        permuted_indices, A_indices, A.BT().blocks_,
                        full_contraction);
                BlockedTensor Abtp = BlockedTensor::build(
                    CoreTensor, A.BT().name() + " permute", A_blocks);
                LabeledBlockedTensor At(Abtp, permuted_indices);
                At = A;
                rhsp.operator*(At);
            }
        }
    }

    // Create intermediate batch tensor for result tensor.
    const Indices &Lt_indices = Lt.indices();
    Indices L_batch_indices;
    L_batch_indices.insert(L_batch_indices.end(),
                           Lt_indices.begin() + batched_size, Lt_indices.end());
    std::vector<std::string> L_batch_blocks =
        BlockedTensor::reduce_rank_block_labels(
            L_batch_indices, Lt_indices, Lt.BT().blocks_, full_contraction);
    BlockedTensor Ltp_batch = BlockedTensor::build(
        CoreTensor, Lt.BT().name() + " batch", L_batch_blocks);
    if (L_batch_blocks.empty())
    {
        Ltp_batch.blocks_[{}] =
            Tensor::build(CoreTensor, Lt.BT().name() + " batch[]", {});
    }
    LabeledBlockedTensor Lt_batch(Ltp_batch, L_batch_indices);

    // Create intermediate batch tensors for tensors to be contracted.
    LabeledBlockedTensorProduct rhs_batch;
    std::map<size_t, BlockedTensor> batch_tensors;
    for (size_t i = 0; i < nterms; ++i)
    {
        const LabeledBlockedTensor &A = rhsp[i];
        if (need_slicing[i][batched_size])
        {
            const Indices &A_indices = A.indices();
            Indices A_batch_indices;
            size_t count = 0;
            for (size_t l = 0; l < batched_size; ++l)
            {
                if (need_slicing[i][l])
                {
                    count++;
                }
            }
            A_batch_indices.insert(A_batch_indices.end(),
                                   A_indices.begin() + count, A_indices.end());
            std::vector<std::string> A_batch_blocks =
                BlockedTensor::reduce_rank_block_labels(
                    A_batch_indices, A_indices, A.BT().blocks_,
                    full_contraction);

            batch_tensors[i] = BlockedTensor::build(
                CoreTensor, A.BT().name() + " batch", A_batch_blocks);
            if (A_batch_blocks.empty())
            {
                batch_tensors[i].blocks_[{}] =
                    Tensor::build(CoreTensor, A.BT().name() + " batch[]", {});
            }
            LabeledBlockedTensor At(batch_tensors[i], A_batch_indices,
                                    A.factor());
            rhs_batch.operator*(At);
        }
        else
        {
            rhs_batch.operator*(A);
        }
    }

    // Figure out all batched indices mo_spaces
    std::vector<std::vector<size_t>> batch_mo_space_keys;
    if (full_contraction)
    {
        batch_mo_space_keys =
            BlockedTensor::label_to_block_keys(batched_indices);
    }
    else
    {
        for (const std::vector<size_t> &uik : unique_indices_keys)
        {
            std::vector<size_t> term_key(uik.begin(),
                                         uik.begin() + batched_size);
            if (std::find(batch_mo_space_keys.begin(),
                          batch_mo_space_keys.end(),
                          term_key) == batch_mo_space_keys.end())
            {
                batch_mo_space_keys.push_back(term_key);
            }
        }
    }

    for (const std::vector<size_t> &batch_keys : batch_mo_space_keys)
    {
        Dimension slicing_dims;
        for (size_t s : batch_keys)
        {
            slicing_dims.push_back(BlockedTensor::mo_space(s).dim());
        }
        Dimension current_batch(batched_size, 0);
        std::vector<std::shared_ptr<BlockedTensor>> inter_AB_tensors(nterms -
                                                                     2);
        std::shared_ptr<std::tuple<bool, std::vector<std::vector<size_t>>,
                                   std::map<std::string, size_t>>>
            expert_info_ptr;
        std::vector<std::shared_ptr<std::tuple<std::vector<std::vector<size_t>>,
                                               std::map<std::string, size_t>>>>
            inter_block_info_ptrs(nterms - 1);

        while (current_batch[0] < slicing_dims[0])
        {

            // Extract result batch
            for (auto &batch_block_key_tensor : Ltp_batch.blocks_)
            {
                const std::vector<size_t> &batch_block_key =
                    batch_block_key_tensor.first;
                std::vector<size_t> corr_perm_block_key(batch_keys);
                corr_perm_block_key.insert(corr_perm_block_key.end(),
                                           batch_block_key.begin(),
                                           batch_block_key.end());
                if (full_contraction or Lt.BT().is_block(corr_perm_block_key))
                {
                    Tensor Lt_batch_block = batch_block_key_tensor.second;
                    Tensor Lt_block = Lt.BT().block(corr_perm_block_key);
                    size_t L_shift = 0, cur_jump = 1;
                    size_t sub_numel = Lt_batch_block.numel();
                    for (int i = batched_size - 1; i >= 0; --i)
                    {
                        L_shift += current_batch[i] * cur_jump;
                        cur_jump *= slicing_dims[i];
                    }
                    L_shift *= sub_numel;
                    std::vector<double> &Lt_batch_data = Lt_batch_block.data();
                    std::vector<double> &Lt_data = Lt_block.data();
                    std::memcpy(Lt_batch_data.data(), Lt_data.data() + L_shift,
                                sub_numel * sizeof(double));
                }
                else
                {
                    batch_block_key_tensor.second.zero();
                }
            }

            for (size_t i = 0; i < nterms; ++i)
            {
                if (need_slicing[i][batched_size])
                {

                    // Extract tensor batch
                    for (auto &batch_block_key_tensor :
                         batch_tensors[i].blocks_)
                    {
                        const std::vector<size_t> &batch_block_key =
                            batch_block_key_tensor.first;
                        std::vector<size_t> corr_perm_block_key;
                        for (size_t l = 0; l < batched_size; ++l)
                        {
                            if (need_slicing[i][l])
                            {
                                corr_perm_block_key.push_back(batch_keys[l]);
                            }
                        }
                        corr_perm_block_key.insert(corr_perm_block_key.end(),
                                                   batch_block_key.begin(),
                                                   batch_block_key.end());
                        if (full_contraction or
                            rhsp[i].BT().is_block(corr_perm_block_key))
                        {
                            Tensor A_batch_block =
                                batch_block_key_tensor.second;
                            Tensor A_block =
                                rhsp[i].BT().block(corr_perm_block_key);
                            size_t cur_shift = 0, cur_jump = 1;
                            size_t sub_numel_A = A_batch_block.numel();
                            for (int l = batched_size - 1; l >= 0; --l)
                            {
                                if (need_slicing[i][l])
                                {
                                    cur_shift += current_batch[l] * cur_jump;
                                    cur_jump *= slicing_dims[l];
                                }
                            }
                            cur_shift *= sub_numel_A;
                            std::vector<double> &A_batch_data =
                                A_batch_block.data();
                            const std::vector<double> &A_data = A_block.data();
                            std::memcpy(A_batch_data.data(),
                                        A_data.data() + cur_shift,
                                        sub_numel_A * sizeof(double));
                        }
                        else
                        {
                            batch_block_key_tensor.second.zero();
                        }
                    }
                }
            }

            // The following code is identical to Lt_batch.contract(rhs_batch,
            // zero_result, add);
            Lt_batch.contract(rhs_batch, zero_result, add, false,
                              inter_AB_tensors, expert_info_ptr,
                              inter_block_info_ptrs);

            // Copy current batch tensor result to the full result tensor
            for (auto &batch_block_key_tensor : Lt_batch.BT().blocks_)
            {
                const std::vector<size_t> &batch_block_key =
                    batch_block_key_tensor.first;
                std::vector<size_t> corr_perm_block_key(batch_keys);
                corr_perm_block_key.insert(corr_perm_block_key.end(),
                                           batch_block_key.begin(),
                                           batch_block_key.end());
                if (full_contraction or Lt.BT().is_block(corr_perm_block_key))
                {
                    Tensor Lt_batch_block = batch_block_key_tensor.second;
                    Tensor Lt_block = Lt.BT().block(corr_perm_block_key);
                    size_t L_shift = 0, cur_jump = 1;
                    size_t sub_numel = Lt_batch_block.numel();
                    for (int i = batched_size - 1; i >= 0; --i)
                    {
                        L_shift += current_batch[i] * cur_jump;
                        cur_jump *= slicing_dims[i];
                    }
                    L_shift *= sub_numel;
                    std::vector<double> &Lt_batch_data = Lt_batch_block.data();
                    std::vector<double> &Lt_data = Lt_block.data();
                    std::memcpy(Lt_data.data() + L_shift, Lt_batch_data.data(),
                                sub_numel * sizeof(double));
                }
            }

            // Determine the indices of next batch
            for (int i = batched_size - 1; i >= 0; --i)
            {
                current_batch[i]++;
                if (current_batch[i] < slicing_dims[i])
                {
                    break;
                }
                else if (i != 0)
                {
                    current_batch[i] = 0;
                }
            }
        }
    }

    // Permute result tensor back
    if (permute_flag)
    {
        (*this) = Lt;
    }
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

pair<double, double> LabeledBlockedTensorProduct::compute_contraction_cost(
    const vector<size_t> &perm,
    const std::vector<std::vector<size_t>> &unique_indices_keys,
    const std::map<std::string, size_t> &index_map, bool full_contraction) const
{
    double cpu_cost_total = 0.0;
    double memory_cost_max = 0.0;
    Indices first = tensors_[perm[0]].indices();
    for (size_t i = 1; i < perm.size(); ++i)
    {
        Indices second = tensors_[perm[i]].indices();
        std::sort(first.begin(), first.end());
        std::sort(second.begin(), second.end());
        Indices common, first_unique, second_unique;

        // cannot use common.begin() here, need to use back_inserter() because
        // common.begin() of an
        // empty vector is not a valid output iterator
        std::set_intersection(first.begin(), first.end(), second.begin(),
                              second.end(), back_inserter(common));
        std::set_difference(first.begin(), first.end(), second.begin(),
                            second.end(), back_inserter(first_unique));
        std::set_difference(second.begin(), second.end(), first.begin(),
                            first.end(), back_inserter(second_unique));

        Indices all = common;
        all.insert(all.end(), first_unique.begin(), first_unique.end());
        all.insert(all.end(), second_unique.begin(), second_unique.end());

        std::vector<std::vector<size_t>> sub_uiks;
        if (full_contraction)
        {
            sub_uiks = BlockedTensor::label_to_block_keys(all);
        }
        else
        {
            size_t max_path = 1;
            for (const auto &index : all)
            {
                max_path *= BlockedTensor::index_to_mo_spaces_[index].size();
            }
            std::set<std::vector<size_t>> set_uiks;
            std::vector<size_t> sub_indices;
            for (const std::string &s : common)
            {
                sub_indices.push_back(index_map.at(s));
            }
            for (const std::string &s : first_unique)
            {
                sub_indices.push_back(index_map.at(s));
            }
            for (const std::string &s : second_unique)
            {
                sub_indices.push_back(index_map.at(s));
            }
            for (const std::vector<size_t> &uik : unique_indices_keys)
            {
                std::vector<size_t> new_uik;
                for (size_t i : sub_indices)
                {
                    new_uik.push_back(uik[i]);
                }
                set_uiks.insert(new_uik);
                if (set_uiks.size() == max_path)
                    break;
            }
            sub_uiks.reserve(set_uiks.size());
            for (const auto &uik : set_uiks)
            {
                sub_uiks.push_back(uik);
            }
        }

        size_t common_max = common.size();
        size_t first_unique_max = common_max + first_unique.size();
        size_t second_unique_max = first_unique_max + second_unique.size();

        double cpu_cost = 0.0, memory_cost = 0.0;
        for (const std::vector<size_t> &uik : sub_uiks)
        {
            size_t j = 0;
            double common_size = 1.0;
            while (j < common_max)
            {
                common_size *= BlockedTensor::mo_space(uik[j++]).dim();
            }
            double first_unique_size = 1.0;
            while (j < first_unique_max)
            {
                first_unique_size *= BlockedTensor::mo_space(uik[j++]).dim();
            }
            double second_unique_size = 1.0;
            while (j < second_unique_max)
            {
                second_unique_size *= BlockedTensor::mo_space(uik[j++]).dim();
            }

            cpu_cost += common_size * first_unique_size * second_unique_size;
            memory_cost += common_size * first_unique_size +
                           common_size * second_unique_size +
                           first_unique_size * second_unique_size;
        }

        Indices stored_indices(first_unique);
        stored_indices.insert(stored_indices.end(), second_unique.begin(),
                              second_unique.end());

        cpu_cost_total += cpu_cost;
        memory_cost_max = std::max({memory_cost_max, memory_cost});

        first = stored_indices;
    }

    std::vector<std::string> vec_str;
    for (size_t i : perm)
    {
        vec_str.push_back(tensors_[i].str());
    }

    return std::make_pair(cpu_cost_total, memory_cost_max);
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

LabeledBlockedTensorBatchedProduct
batched(const string &batched_indices,
        const LabeledBlockedTensorProduct &product)
{
    return LabeledBlockedTensorBatchedProduct(product,
                                              indices::split(batched_indices));
}

} // namespace ambit
