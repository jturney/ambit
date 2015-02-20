#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>

#include <blocked_tensor/blocked_tensor.h>
#include <tensor/indices.h>

namespace tensor {

// Static members of BlockedTensor
std::vector<MOSpace> BlockedTensor::mo_spaces_;
std::map<std::string,size_t> BlockedTensor::name_to_mo_space_;
std::map<std::string,std::vector<size_t>> BlockedTensor::composite_name_to_mo_spaces_;
std::map<std::string,std::vector<size_t>> BlockedTensor::index_to_mo_spaces_;


MOSpace::MOSpace(const std::string& name, const std::string& mo_indices,std::vector<size_t> mos,MOSpaceSpinType spin)
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

BlockedTensor::BlockedTensor() : rank_(0)
{
}

BlockedTensor BlockedTensor::build(TensorType type, const std::string& name, const std::vector<std::string>& blocks)
{
    BlockedTensor newObject;

    newObject.set_name(name);
    newObject.rank_ = 0;

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

        // Set or check the rank
        if (newObject.rank_ > 0){
            if (newObject.rank_ != this_block.size()){
                throw std::runtime_error("Attempting to create the BlockedTensor \"" + name + "\" with nonunique rank.");
            }
        }else{
            newObject.rank_ = this_block.size();
        }
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

size_t BlockedTensor::rank() const
{
    return rank_;
}

void BlockedTensor::set_name(const std::string& name)
{
    name_ = name;
}

bool BlockedTensor::is_valid_block(const std::vector<size_t>& key) const
{
    return (blocks_.count(key) != 0);
}

Tensor BlockedTensor::block(const std::string& indices)
{
//    std::vector<std::string> split_indices = indices::split(indices);
    std::vector<size_t> key;
    for (const std::string& index : indices::split(indices)){
        if (name_to_mo_space_.count(index) != 0){
            key.push_back(name_to_mo_space_[index]);
        }else{
            throw std::runtime_error("Cannot retrieve block " + indices + " of tensor " + name() +
                                     ". The index " + index + " does not indentify a unique space");
        }
    }
    return block(key);
}

Tensor BlockedTensor::block(std::vector<size_t>& key)
{
    if (! is_valid_block(key)){
        std::string msg;
        for (size_t k : key){
            msg += boost::lexical_cast<std::string>(k);
        }
        throw std::runtime_error("Block \"" + msg + "\" is not contained in tensor " + name());
    }
    return blocks_.at(key);
}

const Tensor BlockedTensor::block(std::vector<size_t>& key) const
{
    if (! is_valid_block(key)){
        std::string msg;
        for (size_t k : key){
            msg += boost::lexical_cast<std::string>(k);
        }
        throw std::runtime_error("Block " + msg + " is not contained in tensor " + name());
    }
    return blocks_.at(key);
}

double BlockedTensor::norm(int type) const
{
    if (type == 0) {
        double val = 0.0;
        for (auto block_tensor : blocks_){
            val = std::max(val, std::fabs(block_tensor.second.norm(type)));
        }
        return val;
    } else if (type == 1) {
        double val = 0.0;
        for (auto block_tensor : blocks_){
            val += std::fabs(block_tensor.second.norm(type));
        }
        return val;
    } else if (type == 2) {
        double val = 0.0;
        for (auto block_tensor : blocks_){
            val += std::pow(block_tensor.second.norm(type),2.0);
        }
        return sqrt(val);
    } else {
        throw std::runtime_error("Norm must be 0 (infty-norm), 1 (1-norm), or 2 (2-norm)");
    }
    return 0.0;
}


void BlockedTensor::zero()
{
    for (auto block_tensor : blocks_){
        block_tensor.second.zero();
    }
}

void BlockedTensor::scale(double beta)
{
    for (auto block_tensor : blocks_){
        block_tensor.second.scale(beta);
    }
}

void BlockedTensor::set(double gamma)
{
    for (auto block_tensor : blocks_){
        aligned_vector<double>& data = block_tensor.second.data();
        for (size_t i = 0L; i < data.size(); ++i){
            data[i] = gamma;
        }
    }
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

LabeledBlockedTensor BlockedTensor::operator()(const std::string& indices)
{
    return LabeledBlockedTensor(*this, indices::split(indices));
}

std::vector<std::vector<size_t>> BlockedTensor::label_to_block_keys(const std::vector<std::string>& indices)
{
    // This function takes in the labels used to form a LabeledBlockedTensor and returns the keys
    // to access the corresponding blocks.
    // For example, suppose that indices "i,j,k" are reserved for occupied orbitals ("o") and that index "p"
    // belongs to the composite space of occupied and virtual orbitals ("o" + "v").
    // Then if this function is called with {"i","j","k","p"} it will return the vectors
    // {0,0,0,0} and {0,0,0,1}, which stand for the "oooo" and "ooov" blocks, respectively.
    // The way we proceed is by forming partial vectors that we keep expanding as we
    // process all the indices.

    std::vector<std::vector<size_t>> final_blocks;

    // Loop over indices of this block
    for (const std::string& index : indices){
        std::vector<std::vector<size_t>> partial_blocks;
        // How does this MO space name map to the MOSpace objects contained in mo_spaces_? (e.g. "G" -> {0,1})
        if (index_to_mo_spaces_.count(index) != 0){
            for (size_t mo_space_idx : index_to_mo_spaces_[index]){
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
        }else{
            throw std::runtime_error("Index \"" + index +"\" is not defined.");
        }
        final_blocks = partial_blocks;
    }
    return final_blocks;
}

LabeledBlockedTensor::LabeledBlockedTensor(BlockedTensor BT, const std::vector<std::string>& indices, double factor)
    : BT_(BT), indices_(indices), factor_(factor)
{
    if (BT_.rank() != indices.size())
    throw std::runtime_error("Labeled tensor does not have correct number of indices for underlying tensor's rank");
}

void LabeledBlockedTensor::operator=(const LabeledBlockedTensor &rhs)
{
    std::vector<std::vector<size_t>> rhs_keys = rhs.label_to_block_keys();

    std::vector<size_t> perm = indices::permutation_order(indices_,rhs.indices_);

    // Loop over all keys of the rhs
    for (std::vector<size_t>& rhs_key : rhs_keys){
        // Map the rhs key to the lhs key
        std::vector<size_t> lhs_key;
        for (size_t p : perm){
            lhs_key.push_back(rhs_key[p]);
        }
        // Call LabeledTensor's operation
        Tensor LHS = BT().block(lhs_key);
        const Tensor RHS = rhs.BT().block(rhs_key);

        if (LHS == RHS) throw std::runtime_error("Self assignment is not allowed.");
        if (LHS.rank() != RHS.rank()) throw std::runtime_error("Permuted tensors do not have same rank");
        LHS.permute(RHS,indices_, rhs.indices_, rhs.factor(), 0.0);
    }
}

void LabeledBlockedTensor::operator+=(const LabeledBlockedTensor &rhs)
{
    std::vector<std::vector<size_t>> rhs_keys = rhs.label_to_block_keys();

    std::vector<size_t> perm = indices::permutation_order(indices_,rhs.indices_);

    // Loop over all keys of the rhs
    for (std::vector<size_t>& rhs_key : rhs_keys){
        // Map the rhs key to the lhs key
        std::vector<size_t> lhs_key;
        for (size_t p : perm){
            lhs_key.push_back(rhs_key[p]);
        }
        // Call LabeledTensor's operation
        Tensor LHS = BT().block(lhs_key);
        const Tensor RHS = rhs.BT().block(rhs_key);

        if (LHS == RHS) throw std::runtime_error("Self assignment is not allowed.");
        if (LHS.rank() != RHS.rank()) throw std::runtime_error("Permuted tensors do not have same rank");
        LHS.permute(RHS,indices_, rhs.indices_, rhs.factor(), 1.0);
    }
}

void LabeledBlockedTensor::operator-=(const LabeledBlockedTensor &rhs)
{
    std::vector<std::vector<size_t>> rhs_keys = rhs.label_to_block_keys();

    std::vector<size_t> perm = indices::permutation_order(indices_,rhs.indices_);

    // Loop over all keys of the rhs
    for (std::vector<size_t>& rhs_key : rhs_keys){
        // Map the rhs key to the lhs key
        std::vector<size_t> lhs_key;
        for (size_t p : perm){
            lhs_key.push_back(rhs_key[p]);
        }
        // Call LabeledTensor's operation
        Tensor LHS = BT().block(lhs_key);
        const Tensor RHS = rhs.BT().block(rhs_key);

        if (LHS == RHS) throw std::runtime_error("Self assignment is not allowed.");
        if (LHS.rank() != RHS.rank()) throw std::runtime_error("Permuted tensors do not have same rank");
        LHS.permute(RHS,indices_, rhs.indices_, -rhs.factor(), 1.0);
    }
}

LabeledBlockedTensorProduct LabeledBlockedTensor::operator*(const LabeledBlockedTensor &rhs)
{
    return LabeledBlockedTensorProduct(*this, rhs);
}

//LabeledTensorAddition LabeledTensor::operator+(const LabeledTensor &rhs)
//{
//    return LabeledTensorAddition(*this, rhs);
//}

//LabeledTensorAddition LabeledTensor::operator-(const LabeledTensor &rhs)
//{
//    return LabeledTensorAddition(*this, -rhs);
//}

//namespace {

//LabeledTensor tensor_product_get_temp_AB(const LabeledTensor &A, const LabeledTensor &B)
//{
//    std::vector<Indices> AB_indices = indices::determine_contraction_result(A, B);
//    const Indices &A_fix_idx = AB_indices[1];
//    const Indices &B_fix_idx = AB_indices[2];
//    Dimension dims;
//    Indices indices;

//    for (size_t i = 0; i < A_fix_idx.size(); ++i) {
//        dims.push_back(A.T().dim(i));
//        indices.push_back(A_fix_idx[i]);
//    }
//    for (size_t i = 0; i < B_fix_idx.size(); ++i) {
//        dims.push_back(B.T().dim(i));
//        indices.push_back(B_fix_idx[i]);
//    }

//    Tensor T = Tensor::build(A.T().type(), A.T().name() + " * " + B.T().name(), dims);
//    return T(indices::to_string(indices));
//}

//}

void LabeledBlockedTensor::operator=(const LabeledBlockedTensorProduct &rhs)
{
    contract(rhs,true,true);
}

void LabeledBlockedTensor::operator+=(const LabeledBlockedTensorProduct &rhs)
{
    contract(rhs,false,true);
}

void LabeledBlockedTensor::operator-=(const LabeledBlockedTensorProduct &rhs)
{
    contract(rhs,false,false);
}

void LabeledBlockedTensor::contract(const LabeledBlockedTensorProduct &rhs,
                                    bool zero_result,
                                    bool add)
{
    if (zero_result) BT_.zero();
    // Find the unique indices in the contraction
    size_t nterms = rhs.size();
    std::vector<std::string> unique_indices;
    for (size_t n = 0; n < nterms; ++n){
        for (const std::string& index : rhs[n].indices()){
            unique_indices.push_back(index);
        }
    }
    sort( unique_indices.begin(), unique_indices.end() );
    unique_indices.erase( std::unique( unique_indices.begin(), unique_indices.end() ), unique_indices.end() );

    std::vector<std::vector<size_t>> unique_indices_keys = BlockedTensor::label_to_block_keys(unique_indices);
    std::map<std::string,size_t> index_map;
    {
        size_t k = 0;
        for (const std::string& index : unique_indices){
            index_map[index] = k;
            k++;
        }
    }

    // Setup and perform contractions
    for (const std::vector<size_t>& uik : unique_indices_keys){
        std::vector<size_t> result_key;
        for (const std::string& index : indices()){
            result_key.push_back(uik[index_map[index]]);
        }
        LabeledTensor result(BT().block(result_key),indices(),factor());

        LabeledTensorProduct prod;
        for (size_t n = 0; n < nterms; ++n){
            const LabeledBlockedTensor& lbt = rhs[n];
            std::vector<size_t> term_key;
            for (const std::string& index : lbt.indices()){
                term_key.push_back(uik[index_map[index]]);
            }
            const LabeledTensor term(lbt.BT().block(term_key),lbt.indices(),lbt.factor());
            prod *= term;
        }
        if (add){
            result += prod;
        }else{
            result -= prod;
        }
    }
}

//void LabeledTensor::operator=(const LabeledTensorAddition &rhs)
//{
//    T_.zero();
//    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind) {
//        const LabeledTensor &labeledTensor = rhs[ind];
//        if (T_ == rhs[ind].T()) throw std::runtime_error("Self assignment is not allowed.");
//        if (T_.rank() != rhs[ind].T().rank()) throw std::runtime_error("Permuted tensors do not have same rank");
//        T_.permute(rhs[ind].T(), indices_, rhs[ind].indices(), rhs[ind].factor(), 1.0);
//    }
//}

//void LabeledTensor::operator+=(const LabeledTensorAddition &rhs)
//{
//    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind) {
//        const LabeledTensor &labeledTensor = rhs[ind];
//        if (T_ == rhs[ind].T()) throw std::runtime_error("Self assignment is not allowed.");
//        if (T_.rank() != rhs[ind].T().rank()) throw std::runtime_error("Permuted tensors do not have same rank");
//        T_.permute(rhs[ind].T(), indices_, rhs[ind].indices(), rhs[ind].factor(), 1.0);
//    }
//}

//void LabeledTensor::operator-=(const LabeledTensorAddition &rhs)
//{
//    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind) {
//        const LabeledTensor &labeledTensor = rhs[ind];
//        if (T_ == rhs[ind].T()) throw std::runtime_error("Self assignment is not allowed.");
//        if (T_.rank() != rhs[ind].T().rank()) throw std::runtime_error("Permuted tensors do not have same rank");
//        T_.permute(rhs[ind].T(), indices_, rhs[ind].indices(), -rhs[ind].factor(), 1.0);
//    }
//}

void LabeledBlockedTensor::operator*=(double scale)
{
    std::vector<std::vector<size_t>> keys = label_to_block_keys();

    // Loop over all keys and scale blocks
    for (std::vector<size_t>& key : keys){
        BT_.block(key).scale(scale);
    }
}

void LabeledBlockedTensor::operator/=(double scale)
{
    std::vector<std::vector<size_t>> keys = label_to_block_keys();

    // Loop over all keys and scale blocks
    for (std::vector<size_t>& key : keys){
        BT_.block(key).scale(1.0 / scale);
    }
}

//LabeledTensorDistributive LabeledTensor::operator*(const LabeledTensorAddition &rhs)
//{
//    return LabeledTensorDistributive(*this, rhs);
//}

//void LabeledTensor::operator=(const LabeledTensorDistributive &rhs)
//{
//    T_.zero();

//    for (const LabeledTensor &B : rhs.B()) {
//        *this += const_cast<LabeledTensor &>(rhs.A()) * const_cast<LabeledTensor &>(B);
//    }
//}

//void LabeledTensor::operator+=(const LabeledTensorDistributive &rhs)
//{
//    for (const LabeledTensor &B : rhs.B()) {
//        *this += const_cast<LabeledTensor &>(rhs.A()) * const_cast<LabeledTensor &>(B);
//    }
//}

//void LabeledTensor::operator-=(const LabeledTensorDistributive &rhs)
//{
//    for (const LabeledTensor &B : rhs.B()) {
//        *this -= const_cast<LabeledTensor &>(rhs.A()) * const_cast<LabeledTensor &>(B);
//    }
//}

//LabeledTensorDistributive LabeledTensorAddition::operator*(const LabeledTensor &other)
//{
//    return LabeledTensorDistributive(other, *this);
//}

//LabeledTensorAddition &LabeledTensorAddition::operator*(double scalar)
//{
//    // distribute the scalar to each term
//    for (LabeledTensor &T : tensors_) {
//        T *= scalar;
//    }

//    return *this;
//}

//LabeledTensorAddition &LabeledTensorAddition::operator-()
//{
//    for (LabeledTensor &T : tensors_) {
//        T *= -1.0;
//    }

//    return *this;
//}

//LabeledTensorProduct::operator double() const
//{
//    // Only handles binary expressions.
//    if (size() == 0 || size() > 2)
//        throw std::runtime_error("Conversion operator only supports binary expressions at the moment.");

//    Tensor R = Tensor::build(tensors_[0].T().type(), "R", {});
//    R.contract(
//        tensors_[0].T(),
//        tensors_[1].T(),
//        {},
//        tensors_[0].indices(),
//        tensors_[1].indices(),
//        tensors_[0].factor() * tensors_[1].factor(),
//        1.0);

//    Tensor C = Tensor::build(kCore, "C", {});
//    C.slice(
//        R,
//        {},
//        {});

//    return C.data()[0];
//}

//std::pair<double, double> LabeledTensorProduct::compute_contraction_cost(const std::vector<size_t> &perm) const
//{
//#if 0
//    printf("\n\n  Testing the cost of the contraction pattern: ");
//    for (size_t p : perm)
//        printf("[");
//    for (size_t p : perm) {
//        const LabeledTensor &ti = tensors_[p];
//        printf(" %s] ", indices::to_string(ti.indices()).c_str());
//    }
//#endif

//    std::map<std::string, size_t> indices_to_size;

//    for (const LabeledTensor &ti : tensors_) {
//        const Indices &indices = ti.indices();
//        for (size_t i = 0; i < indices.size(); ++i) {
//            indices_to_size[indices[i]] = ti.T().dim(i);
//        }
//    }

//    double cpu_cost_total = 0.0;
//    double memory_cost_max = 0.0;
//    Indices first = tensors_[perm[0]].indices();
//    for (size_t i = 1; i < perm.size(); ++i) {
//        Indices second = tensors_[perm[i]].indices();
//        std::sort(first.begin(), first.end());
//        std::sort(second.begin(), second.end());
//        Indices common, first_unique, second_unique;

//        // cannot use common.begin() here, need to use back_inserter() because common.begin() of an
//        // empty vector is not a valid output iterator
//        std::set_intersection(first.begin(), first.end(), second.begin(), second.end(), back_inserter(common));
//        std::set_difference(first.begin(), first.end(), second.begin(), second.end(), back_inserter(first_unique));
//        std::set_difference(second.begin(), second.end(), first.begin(), first.end(), back_inserter(second_unique));

//        double common_size = 1.0;
//        for (std::string s : common) common_size *= indices_to_size[s];
//        double first_size = 1.0;
//        for (std::string s : first) first_size *= indices_to_size[s];
//        double second_size = 1.0;
//        for (std::string s : second) second_size *= indices_to_size[s];
//        double first_unique_size = 1.0;
//        for (std::string s : first_unique) first_unique_size *= indices_to_size[s];
//        double second_unique_size = 1.0;
//        for (std::string s : second_unique) second_unique_size *= indices_to_size[s];
//        double result_size = first_unique_size + second_unique_size;


//        std::vector<std::string> stored_indices(first_unique);
//        stored_indices.insert(stored_indices.end(), second_unique.begin(), second_unique.end());

//        double cpu_cost = common_size * result_size;
//        double memory_cost = first_size + second_size + result_size;
//        cpu_cost_total += cpu_cost;
//        memory_cost_max = std::max({memory_cost_max, memory_cost});

//#if 0
//        printf("\n  First indices        : %s", indices::to_string(first).c_str());
//        printf("\n  Second indices       : %s", indices::to_string(second).c_str());

//        printf("\n  Common indices       : %s (%.0f)", indices::to_string(common).c_str(), common_size);
//        printf("\n  First unique indices : %s (%.0f)", indices::to_string(first_unique).c_str(), first_unique_size);
//        printf("\n  Second unique indices: %s (%.0f)", indices::to_string(second_unique).c_str(), second_unique_size);

//        printf("\n  CPU cost for this step    : %f.0", cpu_cost);
//        printf("\n  Memory cost for this step : %f.0 = %f.0 + %f.0 + %f.0",
//               memory_cost,
//               first_size,
//               second_size,
//               result_size);

//        printf("\n  Stored indices       : %s", indices::to_string(stored_indices).c_str());
//#endif

//        first = stored_indices;
//    }

//#if 0
//    printf("\n  Total CPU cost                : %f.0", cpu_cost_total);
//    printf("\n  Maximum memory cost           : %f.0", memory_cost_max);
//#endif

//    return std::make_pair(cpu_cost_total, memory_cost_max);
//}

//LabeledTensorDistributive::operator double() const
//{
//    Tensor R = Tensor::build(A_.T().type(), "R", {});

//    for (size_t ind = 0L; ind < B_.size(); ind++) {

//        R.contract(
//            A_.T(),
//            B_[ind].T(),
//            {},
//            A_.indices(),
//            B_[ind].indices(),
//            B_[ind].factor() * B_[ind].factor(),
//            1.0);
//    }

//    Tensor C = Tensor::build(kCore, "C", {});
//    C.slice(
//        R,
//        {},
//        {});

//    return C.data()[0];
//}


}
