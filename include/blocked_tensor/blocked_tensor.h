#if !defined(TENSOR_INCLUDE_BLOCKED_TENSOR_H)
#define TENSOR_INCLUDE_BLOCKED_TENSOR_H

#include <cstdio>
#include <utility>
#include <vector>
#include <map>
#include <string>

#include <tensor/tensor.h>

namespace tensor {

class LabeledBlockedTensor;
class LabeledBlockedTensorProduct;
class LabeledBlockedTensorAddition;
class LabeledBlockedTensorDistributive;

enum MOSpaceSpinType {AlphaSpin,BetaSpin,NoSpin};

/**
 * Class MOSpace
 **/
class MOSpace
{
public:
    /**
     * Constructor.
     *
     * @param name            The MO space name.
     * @param mo_indices      The MO indices that identify this space.
     * @param mos             The list of MOs that belong to this space.
     * @param spin            The spin of this MO space.
     *
     * Example of use:
     *  // Create a space of alpha occupied orbitals.
     *  MOSpace alpha_occupied("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
     */
    MOSpace(const std::string& name, const std::string& mo_indices, std::vector<size_t> mos,MOSpaceSpinType spin);

    // => Accessors <= //

    /// @return The label of this molecular orbital space
    std::string name() const {return name_;}

    /// @return The indices used to label orbitals in this space
    const std::vector<std::string>& mo_indices() const {return mo_indices_;}

    /// @return The list of molecular orbitals that belong to this space
    const std::vector<size_t>& mos() const {return mos_;}

    /// @return The dimension of the molecular orbital space
    size_t dim() const {return mos_.size();}

    /// @return The spin of this set of molecular orbitals
    MOSpaceSpinType spin() const {return spin_;}

    /// Print information about this molecular orbital space
    void print();
private:
    /// The label of this molecular orbital space
    std::string name_;
    /// The indices used to label orbitals in this space
    std::vector<std::string> mo_indices_;
    /// The list of molecular orbitals that belong to this space
    std::vector<size_t> mos_;
    /// The spin of this set of molecular orbitals
    MOSpaceSpinType spin_;
};


/**
 * Class BlockedTensor
 * Represent a tensor aware of spin and MO spaces.
 * This class holds several tensors, blocked according to spin and MO spaces.
 *
 * Sample usage:
 *  BlockedTensor::add_mo_space("O" ,"i,j,k,l"    ,{0,1,2,3,4},AlphaSpin);
 *  BlockedTensor::add_mo_space("V" ,"a,b,c,d"    ,{7,8,9},AlphaSpin);
 *  BlockedTensor::add_mo_space("I" ,"p,q,r,s,t"  ,{"O","A","V"}); // create a composite space
 *  BlockedTensor::add_mo_space("A" ,"u,v,w,x,y,z",{5,6},AlphaSpin);
 *  BlockedTensor::add_composite_mo_space("H","m,n",{"O","A"}); // BlockedTensor can deal with redundant spaces
 *  BlockedTensor::add_composite_mo_space("P","e,f",{"A","V"});
 *
 *  BlockedTensor T("T","O,O,V,V");
 *  BlockedTensor V("V","O,O,V,V");
 *  E = 0.25 * T("ijab") * V("ijab")
 **/
class BlockedTensor {
    friend class LabeledBlockedTensor;
public:

    // => Constructors <= //

    /// Default constructor.  Does nothing.
    BlockedTensor();

    /**
     * Build a BlockedTensor object
     *
     * @param type            The tensor type enum, one of kCore, kDisk, kDistributed.
     * @param name            The name of the tensor for use in printing.
     * @param blocks          A vector of strings that specify which blocks are contained in this object.
     *
     * Example of use.
     * Here "o","O","v","V" are names of orbital spaces.
     * build(kCore,"T2",{"o,O,v,V"});    // <- creates the alpha-alpha block of the tensor T2 in core
     * build(kDisk,"T1",{"o,v","O,V"}); // <- creates the alpha and beta blocks of the tensor T1 on disk
     */
    static BlockedTensor build(TensorType type, const std::string& name, const std::vector<std::string>& blocks);

    static void add_mo_space(const std::string& name,const std::string& mo_indices,std::vector<size_t> mos,MOSpaceSpinType spin);
    static void add_composite_mo_space(const std::string& name,const std::string& mo_indices,const std::vector<std::string>& subspaces);
    static void reset_mo_spaces();
    static void print_mo_spaces();


    // => Accessors <= //

    /// @return The name of the tensor for use in printing
    std::string name() const;
    /// @return The number of indices in the tensor
    size_t rank() const;
    /// @return The number of blocks
    size_t numblocks() const;

    /// Set the name of the tensor to name
    void set_name(const std::string& name);


    /// Is this block present?
//    bool is_valid_block(const std::vector<std::string>& key);
    /// Is this block present?
    bool is_valid_block(const std::vector<size_t>& key) const;

    /// Return a Tensor object that corresponds to a given orbital class
//    Tensor block(const std::vector<std::string>& key);
    /// Return a Tensor object that corresponds to a given orbital class
    Tensor block(std::vector<size_t>& key);
    const Tensor block(std::vector<size_t>& key) const;
    Tensor block(const std::string& indices);
    std::map<std::vector<size_t>,Tensor>& blocks() {return blocks_;}

    /**
     * Print some tensor information to fh
     * \param level If level = false, just print name and dimensions.  If level = true, print the entire tensor.
     **/
    void print(FILE* fh, bool level = false, const std::string& format = std::string("%11.6f"), int maxcols = 5) const;

    // => Data Access <= //

    /**
     * Returns the raw data vector underlying the tensor object if the
     * underlying tensor object supports a raw data vector. This is only the
     * case if the underlying tensor is of type kCore.
     *
     * Results:
     *  @return data pointer, if tensor object supports it
     **/


//    /// @return The
//    std::map<std::vector<size_t>,Tensor>& blocks() {return blocks_;}

    // => BLAS-Type Tensor Operations <= //

    /**
     * Returns the norm of the tensor
     *
     * Parameters:
     * @param type the type of norm desired:
     *  0 - Infinity-norm, maximum absolute value of elements
     *  1 - One-norm, sum of absolute values of elements
     *  2 - Two-norm, square root of sum of squares
     **/
    double norm(int type = 2) const;

    /**
     * Sets the data of the tensor to zeros.
     * Note: this just drops down to scale(0.0);
     **/
    void zero();

    /**
     * Scales the tensor by scalar beta, e.g.:
     *  C = beta * C
     *
     * Note: If beta is 0.0, a memset is performed rather than a scale to clamp
     * NaNs and other garbage out.
     **/
    void scale(double beta = 0.0);

    /**
     * Set the tensor elemets to gamma, e.g.:
     *  C = gamma
     **/
    void set(double gamma);

//    /**
//     * Copy the data of other into this tensor.
//     * Note: this just drops into slice
//     **/
//    void copy(const BlockedTensor& other);


    // => Iterators <= //

    void iterate(const std::function<void (const std::vector<size_t>&, double&)>& func);
    void citerate(const std::function<void (const std::vector<size_t>&, const double&)>& func) const;
//    void fill(const std::function<void (const std::vector<size_t>&, const double&)>& func) const;

    /// Maps tensor labels ({"i","j","k","p"}) to keys to the block map ({{0,0,0,0},{0,0,0,1}})
    static std::vector<std::vector<size_t>> label_to_block_keys(const std::vector<std::string>& indices);

private:

    std::string name_;
    std::size_t rank_;
    std::map<std::vector<size_t>,Tensor> blocks_;

    /// A vector of MOSpace objects
    size_t add_mo_space(MOSpace mo_space);
    bool map_name_to_mo_space(const std::string& index,size_t mo_space_idx);
    bool map_composite_name_to_mo_spaces(const std::string& index,const std::vector<size_t>& mo_spaces_idx);
    bool map_index_to_mo_spaces(const std::string& index,const std::vector<size_t>& mo_spaces_idx);

    /// @return The n-th MOSpace
    size_t mo_space(size_t n);
    /// @return The MOSpace corresponding to the name of a space
    size_t name_to_mo_space(const std::string& index) ;
    /// @return The MOSpace objects corresponding to the name of a space
    std::vector<size_t>& composite_name_to_mo_spaces(const std::string& index);
    /// @return The MOSpace objects corresponding to an orbital index
    std::vector<size_t>& index_to_mo_spaces(const std::string& index);

    /// A vector of MOSpace objects
    static std::vector<MOSpace> mo_spaces_;
    /// Maps the name of MOSpace (e.g. "o") to the position of the object in the vector mo_spaces_
    static std::map<std::string,size_t> name_to_mo_space_;
    /// Maps the name of a composite orbital space (e.g. "h") to the MOSpace objects that it spans
    static std::map<std::string,std::vector<size_t>> composite_name_to_mo_spaces_;
    /// Maps an orbital index (e.g. "i","j") to the MOSpace objects that contain it
    static std::map<std::string,std::vector<size_t>> index_to_mo_spaces_;

protected:

public:

    // => Operator Overloading API <= //

    LabeledBlockedTensor operator()(const std::string& indices);
};

class LabeledBlockedTensor {

public:
    LabeledBlockedTensor(BlockedTensor T, const std::vector<std::string>& indices, double factor = 1.0);

    double factor() const { return factor_; }
    const Indices& indices() const { return indices_; }
    const BlockedTensor& BT() const { return BT_; }

    LabeledBlockedTensorProduct operator*(const LabeledBlockedTensor& rhs);
    LabeledBlockedTensorAddition operator+(const LabeledBlockedTensor& rhs);
    LabeledBlockedTensorAddition operator-(const LabeledBlockedTensor& rhs);

//    LabeledBlockedTensorDistributive operator*(const LabeledBlockedTensorAddition& rhs);

    /** Copies data from rhs to this sorting the data if needed. */
    void operator=(const LabeledBlockedTensor& rhs);
    void operator+=(const LabeledBlockedTensor& rhs);
    void operator-=(const LabeledBlockedTensor& rhs);

//    void operator=(const LabeledBlockedTensorDistributive& rhs);
//    void operator+=(const LabeledBlockedTensorDistributive& rhs);
//    void operator-=(const LabeledBlockedTensorDistributive& rhs);

    void operator=(const LabeledBlockedTensorProduct& rhs);
    void operator+=(const LabeledBlockedTensorProduct& rhs);
    void operator-=(const LabeledBlockedTensorProduct& rhs);

    void operator=(const LabeledBlockedTensorAddition& rhs);
    void operator+=(const LabeledBlockedTensorAddition& rhs);
    void operator-=(const LabeledBlockedTensorAddition& rhs);

    void operator*=(double scale);
    void operator/=(double scale);

    size_t numdim() const { return indices_.size(); }
//    size_t dim_by_index(const std::string& idx) const;

    // negation
    LabeledBlockedTensor operator-() const {
        return LabeledBlockedTensor(BT_, indices_, -factor_);
    }

private:
    void set(const LabeledBlockedTensor& to);

    std::vector<std::vector<size_t>> label_to_block_keys() const {return BT_.label_to_block_keys(indices_);}

    void contract(const LabeledBlockedTensorProduct &rhs,bool zero_result,bool add);
    void add(const LabeledBlockedTensor &rhs,double alpha,double beta);

    BlockedTensor BT_;
    std::vector<std::string> indices_;
    double factor_;

};

inline LabeledBlockedTensor operator*(double factor, const LabeledBlockedTensor& ti) {
    return LabeledBlockedTensor(ti.BT(), ti.indices(), factor*ti.factor());
};

class LabeledBlockedTensorProduct {

public:
    LabeledBlockedTensorProduct(const LabeledBlockedTensor& A, const LabeledBlockedTensor& B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    size_t size() const { return tensors_.size(); }

    const LabeledBlockedTensor& operator[](size_t i) const { return tensors_[i]; }

    LabeledBlockedTensorProduct& operator*(const LabeledBlockedTensor& other) {
        tensors_.push_back(other);
        return *this;
    }

    // conversion operator
    operator double() const;

private:

    std::vector<LabeledBlockedTensor> tensors_;
};

class LabeledBlockedTensorAddition
{
public:
    LabeledBlockedTensorAddition(const LabeledBlockedTensor& A, const LabeledBlockedTensor& B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    size_t size() const { return tensors_.size(); }

    const LabeledBlockedTensor& operator[](size_t i) const { return tensors_[i]; }

    std::vector<LabeledBlockedTensor>::iterator begin() { return tensors_.begin(); }
    std::vector<LabeledBlockedTensor>::const_iterator begin() const { return tensors_.begin(); }

    std::vector<LabeledBlockedTensor>::iterator end() { return tensors_.end(); }
    std::vector<LabeledBlockedTensor>::const_iterator end() const { return tensors_.end(); }

    LabeledBlockedTensorAddition& operator+(const LabeledBlockedTensor& other) {
        tensors_.push_back(other);
        return *this;
    }

    LabeledBlockedTensorAddition& operator-(const LabeledBlockedTensor& other) {
        tensors_.push_back(-other);
        return *this;
    }

    LabeledBlockedTensorDistributive operator*(const LabeledBlockedTensor& other);

    LabeledBlockedTensorAddition& operator*(double scalar);

    // negation
    LabeledBlockedTensorAddition& operator-();

private:

    // This handles cases like T("ijab")
    std::vector<LabeledBlockedTensor> tensors_;

};

inline LabeledBlockedTensorAddition operator*(double factor, const LabeledBlockedTensorAddition& ti) {
    LabeledBlockedTensorAddition ti2 = ti;
    return ti2 * factor;
}

// Is responsible for expressions like D * (J - K) --> D*J - D*K
class LabeledBlockedTensorDistributive
{
public:
    LabeledBlockedTensorDistributive(const LabeledBlockedTensor& A, const LabeledBlockedTensorAddition& B)
            : A_(A), B_(B)
    {}

    const LabeledBlockedTensor& A() const { return A_; }
    const LabeledBlockedTensorAddition& B() const { return B_; }

    // conversion operator
    operator double() const;

private:

    const LabeledBlockedTensor& A_;
    const LabeledBlockedTensorAddition& B_;

};

}

#endif

