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

//void BlockedTensor::add_primitive_mo_space(std::string label, std::string indices, std::vector<size_t> mo, MOSetSpinType spin)
//{
//    // Input format 1: a series of characters ""
//    std::vector<std::string> i_vec;
//    for (char c : indices){
//        i_vec.push_back(std::string(1,c));
//    }
//    MOSet ms(label, i_vec, mo, spin);
//    // Add a map from the MOSet label to the object
//    label_to_prim_set_[label] = prim_set_.size();
//    prim_set_.push_back(ms);
//    // Add a map from the label of this set to the list of all its subsets
//    label_to_prim_set_label_[label] = {label};
//    // Add a map from the indices of this set to the list of all its subsets
//    for (std::string index : i_vec){
//        indices_to_prim_set_label_[index] = {label};
//    }
//}

//void BlockedTensor::add_composite_mo_space(const std::string& label,const std::string& indices,const std::vector<std::string>& subspaces)
//{
//    // Input format 1: a series of characters ""
//    std::vector<std::string> i_vec;
//    for (char c : indices){
//        i_vec.push_back(std::string(1,c));
//    }
//    // Add a map from the label of this set to the list of all its subsets
//    for (std::string subspace : subspaces){
//        if (label_to_prim_set_.count(subspace) == 0){
//            outfile->Printf("\n\nERROR in BlockedTensor::add_composite_mo_space:");
//            outfile->Printf("\nSubspace [%s] is not defined",subspace.c_str());
//            outfile->Flush();
//            exit(1);
//        }
//    }
//    label_to_prim_set_label_[label] = subspaces;
//    // Add a map from the indices of this set to the list of all its subsets
//    for (std::string index : i_vec){
//        indices_to_prim_set_label_[index] = subspaces;
//    }
//}

BlockedTensor::BlockedTensor()
{
}

BlockedTensor BlockedTensor::build(TensorType type, const std::string& name, const std::vector<std::string>& blocks)
{
    BlockedTensor newObject;

    std::vector<std::vector<int>> tensor_blocks;

    // This algorithm takes a vector of strings that define the blocks of this tensor and unpacks them.
    // This may require taking composite spaces ("G" = {"O","V"}) and expanding the string
    // "G,G" -> {"O,O" , "O,V" , "V,O" , "V,V"}.
    // The way we proceed is by forming partial strings that we keep expanding as we
    // process all the indices.

    // Loop over blocks given to us
    for (const std::string& this_block: blocks){
        std::vector<std::vector<int>> final_blocks;

        std::vector<std::string> this_block_vec = indices::split(this_block);
        // Loop over indices of this block
        for (std::string mo_space_name : this_block_vec){
            std::vector<std::vector<int>> partial_blocks;
            // How does this MO space name map to the MOSpace objects contained in mo_spaces_? (e.g. "G" -> {0,1})
            for (int mo_space_idx : composite_name_to_mo_spaces_[mo_space_name]){
                // Special case
                if(final_blocks.size() == 0){
                    partial_blocks.push_back({mo_space_idx});
                }else{
                    // Add each this primitive set to all the partial block labels
                    for (std::vector<int>& block : final_blocks){
                        std::vector<int> new_block(block);
                        new_block.push_back(mo_space_idx);
                        partial_blocks.push_back(new_block);
                    }
                }
            }
            final_blocks = partial_blocks;
        }
        for (std::vector<int>& block : final_blocks) tensor_blocks.push_back(block);
    }

    // Create the blocks
    printf("\n  Tensor name: %s",name.c_str());
    for (std::vector<int>& this_block : tensor_blocks){
        printf("\n  Creating block:");
        for (int ms : this_block){
               printf(" %s",mo_spaces_[ms].name().c_str());
        }
//        if(print_level_ > 1){
//            outfile->Printf("\n    - Block %5d: [%s]",++k,boost::algorithm::join(block, ",").c_str());
//        }
//        std::vector<size_t> dims;
//        for (std::string set_label : block){
//            size_t dim = prim_set_[label_to_prim_set_[set_label]].dim();
//            dims.push_back(dim);
//        }
//        blocks_[block] = SharedTensor(new Tensor(label_,dims));
    }

//    newObject.tensor_.reset(new DiskTensorImpl(name, dims));

    return newObject;
}

}
