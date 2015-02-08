#if !defined(TENSOR_INCLUDE_BLOCKED_TENSOR_H)
#define TENSOR_INCLUDE_BLOCKED_TENSOR_H

#include <cstdio>
#include <utility>
#include <vector>
#include <map>
#include <string>

#include "tensor.h"

namespace tensor {

class LabeledBlockedTensor;

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
     * @param name            The MO space label.
     * @param mo_indices      The MO indices that identify this space.
     * @param mos             The list of MOs that belong to this space.
     * @param spin            The spin of this MO space.
     *
     * Example of use:
     *  // Create a space of alpha occupied orbitals.
     *  MOSpace alpha_occupied("o","i,j,k,l",{0,1,2,3,4},Alpha);
     */
    MOSpace(std::string name,std::string mo_indices,std::vector<size_t> mos,MOSpaceSpinType spin);

    // => Accessors <= //

    /// @return The label of this molecular orbital space
    std::string name() const {return name_;}

    /// @return The indices used to label orbitals in this space
    std::vector<std::string> mo_indices() const {return mo_indices_;}

    /// @return The list of molecular orbitals that belong to this space
    std::vector<size_t> mos() const {return mos_;}

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
 * Represent a tensor aware of spin and MO spaces
 * This class holds several tensors, blocked according to spin and MO spaces.
 *
 * Sample usage:
 *  BlockedTensor::add_mo_space("O" ,"i,j,k,l"    ,{0,1,2,3,4});
 *  BlockedTensor::add_mo_space("V" ,"a,b,c,d"    ,{7,8,9});
 *  BlockedTensor::add_mo_space("I" ,"p,q,r,s,t"  ,{"O","A","V"}); // create a composite space
 *  BlockedTensor::add_mo_space("A" ,"u,v,w,x,y,z",{5,6});
 *  BlockedTensor::add_mo_space("H" ,"m,n"        ,{"O","A"});
 *  BlockedTensor::add_mo_space("P" ,"e,f"        ,{"A","V"});
 *
 *  BlockedTensor T("T","O,O,V,V");
 *  BlockedTensor V("V","O,O,V,V");
 *  E = 0.25 * T("ijab") * V("ijab")
 **/
class BlockedTensor {

public:

    // => Constructors <= //

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

    BlockedTensor();

    // => Accessors <= //

    /// @return The name of the tensor for use in printing
    std::string name() const;
    /// @return The number of blocks
    size_t numblocks() const;

    /// Set the name of the tensor to name
    void set_name(const std::string& name);

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
     * This routine is intended to facilitate rapid filling of data into a
     * kCore buffer tensor, following which the user may stripe the buffer
     * tensor into a kDisk or kDistributed tensor via slice operations.
     *
     * If a vector is successfully returned, it points to the unrolled data of
     * the tensor, with the right-most dimensions running fastest and left-most
     * dimensions running slowest.
     *
     * Example successful use case:
     *  Tensor A = Tensor::build(kCore, "A3", {4,5,6});
     *  std::vector<double>& Av = A.data();
     *  double* Ap = Av.data(); // In case the raw pointer is needed
     *  In this case, Av[0] = A(0,0,0), Av[1] = A(0,0,1), etc.
     *
     *  Tensor B = Tensor::build(kDisk, "B3", {4,5,6});
     *  std::vector<double>& Bv = B.data(); // throws
     *
     * Results:
     *  @return data pointer, if tensor object supports it
     **/
    /// Return a Tensor object that corresponds to a given orbital class
    Tensor block(const std::string& block_indices);

    /// Is this block present?
    bool is_block(const std::string& block_indices);

    /// @return The
    std::map<std::vector<size_t>,Tensor>& blocks() {return blocks_;}

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
     * Copy the data of other into this tensor.
     * Note: this just drops into slice
     **/
    void copy(const BlockedTensor& other);

    /**
     * Perform the contraction:
     *  C(Cinds) = alpha * A(Ainds) * B(Binds) + beta * C(Cinds)
     *
     * Note: Most users should instead use the operator overloading
     * routines, e.g.,
     *  C2("ij") += 0.5 * A2("ik") * B2("jk");
     *
     * Parameters:
     *  @param A The left-side factor tensor, e.g., A2
     *  @param B The right-side factor tensor, e.g., B2
     *  @param Cinds The indices of tensor C, e.g., "ij"
     *  @param Ainds The indices of tensor A, e.g., "ik"
     *  @param Binds The indices of tensor B, e.g., "jk"
     *  @param alpha The scale applied to the product A*B, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     **/
    void contract(
        const BlockedTensor& A,
        const BlockedTensor& B,
        const std::vector<std::string>& Cinds,
        const std::vector<std::string>& Ainds,
        const std::vector<std::string>& Binds,
        double alpha = 1.0,
        double beta = 0.0);

private:

    std::string name_;
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


public:

    // => Operator Overloading API <= //

    LabeledBlockedTensor operator()(const std::string& indices);
};

class LabeledBlockedTensor {

public:
    LabeledBlockedTensor(BlockedTensor T, const std::vector<std::string>& indices, double factor = 1.0);

    double factor() const { return factor_; }
    const Indices& indices() const { return indices_; }
    const BlockedTensor& T() const { return BT_; }

//    LabeledBlockedTensorProduct operator*(const LabeledBlockedTensor& rhs);
//    LabeledBlockedTensorAddition operator+(const LabeledBlockedTensor& rhs);
//    LabeledBlockedTensorAddition operator-(const LabeledBlockedTensor& rhs);

//    LabeledBlockedTensorDistributive operator*(const LabeledBlockedTensorAddition& rhs);

//    /** Copies data from rhs to this sorting the data if needed. */
//    void operator=(const LabeledTensor& rhs);
//    void operator+=(const LabeledTensor& rhs);
//    void operator-=(const LabeledTensor& rhs);

//    void operator=(const LabeledTensorDistributive& rhs);
//    void operator+=(const LabeledTensorDistributive& rhs);
//    void operator-=(const LabeledTensorDistributive& rhs);

//    void operator=(const LabeledTensorProduct& rhs);
//    void operator+=(const LabeledTensorProduct& rhs);
//    void operator-=(const LabeledTensorProduct& rhs);

//    void operator=(const LabeledTensorAddition& rhs);
//    void operator+=(const LabeledTensorAddition& rhs);
//    void operator-=(const LabeledTensorAddition& rhs);

//    void operator*=(double scale);
//    void operator/=(double scale);

//    size_t numdim() const { return indices_.size(); }
//    size_t dim_by_index(const std::string& idx) const;

//    // negation
//    LabeledTensor operator-() const {
//        return LabeledTensor(T_, indices_, -factor_);
//    }

private:

    void set(const LabeledBlockedTensor& to);

    BlockedTensor BT_;
    std::vector<std::string> indices_;
    double factor_;

};

//inline LabeledTensor operator*(double factor, const LabeledTensor& ti) {
//    return LabeledTensor(ti.T(), ti.indices(), factor*ti.factor());
//};

//class LabeledTensorProduct {

//public:
//    LabeledTensorProduct(const LabeledTensor& A, const LabeledTensor& B)
//    {
//        tensors_.push_back(A);
//        tensors_.push_back(B);
//    }

//    size_t size() const { return tensors_.size(); }

//    const LabeledTensor& operator[](size_t i) const { return tensors_[i]; }

//    LabeledTensorProduct& operator*(const LabeledTensor& other) {
//        tensors_.push_back(other);
//        return *this;
//    }

//    // conversion operator
//    operator double() const;

//    std::pair<double, double> compute_contraction_cost(const std::vector<size_t>& perm) const;

//private:

//    std::vector<LabeledTensor> tensors_;
//};

//class LabeledTensorAddition
//{
//public:
//    LabeledTensorAddition(const LabeledTensor& A, const LabeledTensor& B)
//    {
//        tensors_.push_back(A);
//        tensors_.push_back(B);
//    }

//    size_t size() const { return tensors_.size(); }

//    const LabeledTensor& operator[](size_t i) const { return tensors_[i]; }

//    std::vector<LabeledTensor>::iterator begin() { return tensors_.begin(); }
//    std::vector<LabeledTensor>::const_iterator begin() const { return tensors_.begin(); }

//    std::vector<LabeledTensor>::iterator end() { return tensors_.end(); }
//    std::vector<LabeledTensor>::const_iterator end() const { return tensors_.end(); }

//    LabeledTensorAddition& operator+(const LabeledTensor& other) {
//        tensors_.push_back(other);
//        return *this;
//    }

//    LabeledTensorAddition& operator-(const LabeledTensor& other) {
//        tensors_.push_back(-other);
//        return *this;
//    }

//    LabeledTensorDistributive operator*(const LabeledTensor& other);

//    LabeledTensorAddition& operator*(double scalar);

//    // negation
//    LabeledTensorAddition& operator-();

//private:

//    // This handles cases like T("ijab")
//    std::vector<LabeledTensor> tensors_;

//};

//inline LabeledTensorAddition operator*(double factor, const LabeledTensorAddition& ti) {
//    LabeledTensorAddition ti2 = ti;
//    return ti2 * factor;
//}

//// Is responsible for expressions like D * (J - K) --> D*J - D*K
//class LabeledTensorDistributive
//{
//public:
//    LabeledTensorDistributive(const LabeledTensor& A, const LabeledTensorAddition& B)
//            : A_(A), B_(B)
//    {}

//    const LabeledTensor& A() const { return A_; }
//    const LabeledTensorAddition& B() const { return B_; }

//    // conversion operator
//    operator double() const;

//private:

//    const LabeledTensor& A_;
//    const LabeledTensorAddition& B_;

//};

}

#endif

