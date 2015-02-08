#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>

#include <tensor/blocked_tensor.h>
#include "indices.h"

namespace tensor {

// Static members
std::vector<MOSpace> BlockedTensor::mo_spaces_;
std::map<std::string,size_t> BlockedTensor::name_to_mo_space_;
std::map<std::string,std::vector<size_t>> BlockedTensor::composite_name_to_mo_spaces_;
std::map<std::string,std::vector<size_t>> BlockedTensor::index_to_mo_spaces_;


MOSpace::MOSpace(std::string name,std::string mo_indices,std::vector<size_t> mos,MOSpaceSpinType spin)
    : name_(name), mo_indices_(indices::split(mo_indices)), mos_(mos), spin_(spin)
{}

void MOSpace::print()
{
    std::vector<std::string> mo_list;
    for (size_t i : mos_){
        mo_list.push_back(boost::lexical_cast<std::string>(i));
    }
    printf("\n  Orbital Space \"%s\"\n  MO Indices: {%s}\n  MO List: (%s)\n",name_.c_str(),
           boost::algorithm::join(mo_indices_, ",").c_str(),
           boost::algorithm::join(mo_list, ",").c_str());
}


void BlockedTensor::add_mo_space(const std::string& name,const std::string& mo_indices,std::vector<size_t> mos,MOSpaceSpinType spin)
{
    if (name_to_mo_space_.count(name) != 0){
        throw std::runtime_error("The MO space \"" + name + "\" is already defined.");
    }

    size_t mo_space_idx = mo_spaces_.size();

    MOSpace ms(name,mo_indices,mos,spin);
    // Add the MOSpace object
    mo_spaces_.push_back(ms);

    // Link the name to the mo_space_ vector
    name_to_mo_space_[name] = mo_space_idx;

    // Link the composite name to the mo_space_ vector
    composite_name_to_mo_spaces_[name] = {mo_space_idx};

    // Link the indices to the mo_space_
    for (const std::string& mo_index : indices::split(mo_indices)){
        if (index_to_mo_spaces_.count(mo_index) == 0){
            index_to_mo_spaces_[mo_index] = {mo_space_idx};
        }else{
            throw std::runtime_error("The MO index \"" + mo_index + "\" is already defined.");
        }
    }
}

void BlockedTensor::add_composite_mo_space(const std::string& name,const std::string& mo_indices,const std::vector<std::string>& subspaces)
{
    if (name_to_mo_space_.count(name) != 0){
        throw std::runtime_error("The MO space \"" + name + "\" is already defined.");
    }

    std::vector<size_t> simple_spaces;
    for (std::string subspace : subspaces){
        // Is this simple MO space in our list of spaces?
        if (name_to_mo_space_.count(subspace) == 0){
            throw std::runtime_error("The simple MO space \"" + subspace + "\" is not defined.");
        }else{
            simple_spaces.push_back(name_to_mo_space_[subspace]);
        }
    }
    composite_name_to_mo_spaces_[name] = simple_spaces;

    // Link the indices to the mo_space_
    for (const std::string& mo_index : indices::split(mo_indices)){
        if (index_to_mo_spaces_.count(mo_index) == 0){
            index_to_mo_spaces_[mo_index] = simple_spaces;
        }else{
            throw std::runtime_error("The MO index \"" + mo_index + "\" is already defined.");
        }
    }
}

void BlockedTensor::print_mo_spaces()
{
    printf("\n  List of Molecular Orbital Spaces:");
    for (size_t ms = 0; ms < mo_spaces_.size(); ++ms){
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

BlockedTensor::BlockedTensor()
{
}

BlockedTensor BlockedTensor::build(TensorType type, const std::string& name, const std::vector<std::string>& blocks)
{
    BlockedTensor newObject;

    newObject.set_name(name);

    std::vector<std::vector<size_t>> tensor_blocks;

    // This algorithm takes a vector of strings that define the blocks of this tensor and unpacks them.
    // This may require taking composite spaces ("G" = {"O","V"}) and expanding the string
    // "G,G" -> {"O,O" , "O,V" , "V,O" , "V,V"}.
    // The way we proceed is by forming partial strings that we keep expanding as we
    // process all the indices.

    // Transform composite indices into simple indices
    for (const std::string& this_block: blocks){
        std::vector<std::vector<size_t>> final_blocks;

        std::vector<std::string> this_block_vec = indices::split(this_block);
        // Loop over indices of this block
        for (std::string mo_space_name : this_block_vec){
            std::vector<std::vector<size_t>> partial_blocks;
            // How does this MO space name map to the MOSpace objects contained in mo_spaces_? (e.g. "G" -> {0,1})
            for (size_t mo_space_idx : composite_name_to_mo_spaces_[mo_space_name]){
                // Special case
                if(final_blocks.size() == 0){
                    partial_blocks.push_back({mo_space_idx});
                }else{
                    // Add each this primitive set to all the partial block labels
                    for (std::vector<size_t>& block : final_blocks){
                        std::vector<size_t> new_block(block);
                        new_block.push_back(mo_space_idx);
                        partial_blocks.push_back(new_block);
                    }
                }
            }
            final_blocks = partial_blocks;
        }
        for (std::vector<size_t>& block : final_blocks) tensor_blocks.push_back(block);
    }

    // Create the blocks
    for (std::vector<size_t>& this_block : tensor_blocks){
        // Grab the dims
        std::vector<size_t> dims;
        for (size_t ms : this_block){
            size_t dim = mo_spaces_[ms].dim();
            dims.push_back(dim);
        }
        // Grab the orbital spaces names
        std::string mo_names;
        for (size_t ms : this_block){
            mo_names += mo_spaces_[ms].name();
        }
        newObject.blocks_[this_block] = Tensor::build(type,name + "[" + mo_names + "]",dims);
    }

//    newObject.print(stdout);
    return newObject;
}

size_t BlockedTensor::numblocks() const
{
    return blocks_.size();
}

std::string BlockedTensor::name() const
{
    return name_;
}

void BlockedTensor::set_name(const std::string& name)
{
    name_ = name;
}


void BlockedTensor::print(FILE *fh, bool level, std::string const &format, int maxcols) const
{
    fprintf(fh, "  ## Blocked Tensor %s ##\n\n", name().c_str());
    fprintf(fh, "  Number of blocks = %zu\n", numblocks());
    for (auto kv : blocks_){
        fprintf(fh, "\n");
        kv.second.print(fh, level, format, maxcols);
    }
}

}
