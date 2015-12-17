#if !defined(TENSOR_INCLUDE_BLOCKED_TENSOR_H)
#define TENSOR_INCLUDE_BLOCKED_TENSOR_H

#include <cstdio>
#include <utility>
#include <vector>
#include <map>
#include <string>

#include <ambit/tensor.h>

namespace ambit
{

class LabeledBlockedTensor;
class LabeledBlockedTensorProduct;
class LabeledBlockedTensorAddition;
class LabeledBlockedTensorDistributive;

enum SpinType
{
    AlphaSpin,
    BetaSpin,
    NoSpin
};

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
    MOSpace(const std::string &name, const std::string &mo_indices,
            std::vector<size_t> mos, SpinType spin);

    /**
     * Constructor.
     *
     * @param name            The MO space name.
     * @param mo_indices      The MO indices that identify this space.
     * @param mos_spin        The list of pairs (MO,spin) for all the spin
     * orbitals that belong to this space.
     *
     * Example of use:
     *  // Create a space of alpha and beta occupied orbitals.
     *  MOSpace
     * alpha_occupied("o","i,j,k,l",{(0,AlphaSpin),(0,BetaSpin),(1,AlphaSpin),(2,BetaSpin)});
     */
    MOSpace(const std::string &name, const std::string &mo_indices,
            std::vector<std::pair<size_t, SpinType>> mos_spin);

    // => Accessors <= //

    /// @return The label of this molecular orbital space
    std::string name() const { return name_; }

    /// @return The indices used to label orbitals in this space
    const std::vector<std::string> &mo_indices() const { return mo_indices_; }

    /// @return The list of molecular orbitals that belong to this space
    const std::vector<size_t> &mos() const { return mos_; }

    /// @return The dimension of the molecular orbital space
    size_t dim() const { return mos_.size(); }

    /// @return The spin of this set of molecular orbitals
    std::vector<SpinType> spin() const { return spin_; }

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
    std::vector<SpinType> spin_;
};

/**
 * Class BlockedTensor
 * Represent a tensor aware of spin and MO spaces.
 * This class holds several tensors, blocked according to spin and MO spaces.
 *
 * Sample usage:
 *  BlockedTensor::add_mo_space("O" ,"i,j,k,l"    ,{0,1,2,3,4},AlphaSpin);
 *  BlockedTensor::add_mo_space("V" ,"a,b,c,d"    ,{7,8,9},AlphaSpin);
 *  BlockedTensor::add_mo_space("I" ,"p,q,r,s,t"  ,{"O","A","V"}); // create a
 *composite space
 *  BlockedTensor::add_mo_space("A" ,"u,v,w,x,y,z",{5,6},AlphaSpin);
 *  BlockedTensor::add_composite_mo_space("H","m,n",{"O","A"}); // BlockedTensor
 *can deal with redundant spaces
 *  BlockedTensor::add_composite_mo_space("P","e,f",{"A","V"});
 *
 *  BlockedTensor T("T","O,O,V,V");
 *  BlockedTensor V("V","O,O,V,V");
 *  E = 0.25 * T("ijab") * V("ijab")
 **/
class BlockedTensor
{
    friend class LabeledBlockedTensor;

  public:
    // => Constructors <= //

    /// Default constructor.  Does nothing.
    BlockedTensor();

    /**
     * Build a BlockedTensor object
     *
     * @param type            The tensor type enum, one of CoreTensor, DiskTensor,
     * DistributedTensor.
     * @param name            The name of the tensor for use in printing.
     * @param blocks          A vector of strings that specify which blocks are
     * contained in this object.
     *
     * Example of use.
     * Here "o","O","v","V" are names of orbital spaces.
     * build(CoreTensor,"T2",{"o,O,v,V"});    // <- creates the alpha-alpha block of
     * the tensor T2 in core
     * build(DiskTensor,"T1",{"o,v","O,V"}); // <- creates the alpha and beta blocks
     * of the tensor T1 on disk
     */
    static BlockedTensor build(TensorType type, const std::string &name,
                               const std::vector<std::string> &blocks);

    static void add_mo_space(const std::string &name,
                             const std::string &mo_indices,
                             std::vector<size_t> mos, SpinType spin);
    static void add_mo_space(const std::string &name,
                             const std::string &mo_indices,
                             std::vector<std::pair<size_t, SpinType>> mo_spin);
    static void
    add_composite_mo_space(const std::string &name,
                           const std::string &mo_indices,
                           const std::vector<std::string> &subspaces);
    static void reset_mo_spaces();
    static void print_mo_spaces();

    static void set_expert_mode(bool mode) { expert_mode_ = mode; }

    // => Accessors <= //

    /// @return The name of the tensor for use in printing
    std::string name() const;
    /// @return The number of indices in the tensor
    size_t rank() const;
    /// @return The number of blocks
    size_t numblocks() const;

    /// Set the name of the tensor to name
    void set_name(const std::string &name);

    /// @return Does this Tensor point to the same underlying tensor as Tensor
    /// other?
    bool operator==(const BlockedTensor &other) const;
    /// @return !Does this Tensor point to the same underlying tensor as Tensor
    /// other?
    bool operator!=(const BlockedTensor &other) const;

    /**
    * Print some tensor information to fh
    * \param level If level = false, just print name and dimensions.  If level =
    *true, print the entire tensor.
    **/
    void print(FILE *fh = stdout, bool level = true,
               const std::string &format = std::string("%11.6f"),
               int maxcols = 5) const;

    // => Data Access <= //

    /// @return a list of labels of the blocks contained in this object (e.g.
    /// {"ccaa","cCaA",...})
    std::vector<std::string> &block_labels() { return block_labels_; }
    const

        /**
         * Returns a map with the block key and the corresponding tensor.
         *
         * Results:
         *  @return a key/Tensor map
         **/
        std::map<std::vector<size_t>, Tensor> &
        blocks()
    {
        return blocks_;
    }

    /// Is this block present?
    bool is_block(const std::string &indices) const;
    /// Is this block present?
    bool is_block(const std::vector<size_t> &key) const;

    /// Return a Tensor object that corresponds to a given orbital class
    Tensor block(const std::vector<size_t> &key);
    /// Return a constant Tensor object that corresponds to a given orbital
    /// class
    const Tensor block(const std::vector<size_t> &key) const;
    /// Return a Tensor object that corresponds to a given block key
    Tensor block(const std::string &indices);

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

    /**
     * Copy the data of other into this blocked tensor:
     *  C() = other()
     * Note: this just drops into slice
     *
     * Parameters:
     *  @param other the blocked tensor to copy data from
     *
     * Results
     *  C is the current bocked tensor, whose data is overwritten
     **/
    //    void copy(const BlockedTensor& other);

    // => Iterators <= //

    /**
     * Iterate overall all elements of all blocks.  The iterator provides access
     *to the
     * value of the tensor elements, the MO indices, and spin values.
     **/
    void iterate(const std::function<void(const std::vector<size_t> &,
                                          const std::vector<SpinType> &,
                                          double &)> &func);
    /**
     * Iterate overall all elements of all blocks.  The iterator provides
     *constant access to the
     * value of the tensor elements, the MO indices, and spin values.
     **/
    void citerate(const std::function<void(const std::vector<size_t> &,
                                           const std::vector<SpinType> &,
                                           const double &)> &func) const;

    /// Maps tensor labels ({"i","j","k","p"}) to keys to the block map
    /// ({{0,0,0,0},{0,0,0,1}})
    static std::vector<std::vector<size_t>>
    label_to_block_keys(const std::vector<std::string> &indices);

  private:
    std::string name_;
    std::size_t rank_;
    std::vector<std::string> block_labels_;
    std::map<std::vector<size_t>, Tensor> blocks_;

    /// A vector of MOSpace objects
    size_t add_mo_space(MOSpace mo_space);
    bool map_name_to_mo_space(const std::string &index, size_t mo_space_idx);
    bool
    map_composite_name_to_mo_spaces(const std::string &index,
                                    const std::vector<size_t> &mo_spaces_idx);
    bool map_index_to_mo_spaces(const std::string &index,
                                const std::vector<size_t> &mo_spaces_idx);
    static std::vector<size_t> indices_to_key(const std::string &indices);

    /// @return The n-th MOSpace
    static MOSpace mo_space(size_t n) { return mo_spaces_[n]; }
    /// @return The MOSpace corresponding to the name of a space
    size_t name_to_mo_space(const std::string &index);
    /// @return The MOSpace objects corresponding to the name of a space
    std::vector<size_t> &composite_name_to_mo_spaces(const std::string &index);
    /// @return The MOSpace objects corresponding to an orbital index
    std::vector<size_t> &index_to_mo_spaces(const std::string &index);

    // => Static Class Data <= //

    /// A vector of MOSpace objects
    static std::vector<MOSpace> mo_spaces_;
    /// Maps the name of MOSpace (e.g. "o") to the position of the object in the
    /// vector mo_spaces_
    static std::map<std::string, size_t> name_to_mo_space_;
    /// Maps the name of a composite orbital space (e.g. "h") to the MOSpace
    /// objects that it spans
    static std::map<std::string, std::vector<size_t>>
        composite_name_to_mo_spaces_;
    /// Maps an orbital index (e.g. "i","j") to the MOSpace objects that contain
    /// it
    static std::map<std::string, std::vector<size_t>> index_to_mo_spaces_;
    /// Enables expert mode, which overides some default error checking
    static bool expert_mode_;

  public:
    /// @return Is BlockedTensor using "expert mode"?
    static bool expert_mode() { return expert_mode_; }

  protected:
  public:
    // => Operator Overloading API <= //

    LabeledBlockedTensor operator()(const std::string &indices);
    LabeledBlockedTensor operator[](const std::string &indices);
};

class LabeledBlockedTensor
{

  public:
    LabeledBlockedTensor(BlockedTensor T,
                         const std::vector<std::string> &indices,
                         double factor = 1.0);

    double factor() const { return factor_; }
    const Indices &indices() const { return indices_; }
    const BlockedTensor &BT() const { return BT_; }
    std::string str() const;

    LabeledBlockedTensorProduct operator*(const LabeledBlockedTensor &rhs);
    LabeledBlockedTensorAddition operator+(const LabeledBlockedTensor &rhs);
    LabeledBlockedTensorAddition operator-(const LabeledBlockedTensor &rhs);

    LabeledBlockedTensorDistributive
    operator*(const LabeledBlockedTensorAddition &rhs);

    /** Copies data from rhs to this sorting the data if needed. */
    void operator=(const LabeledBlockedTensor &rhs);
    void operator+=(const LabeledBlockedTensor &rhs);
    void operator-=(const LabeledBlockedTensor &rhs);

    void operator=(const LabeledBlockedTensorDistributive &rhs);
    void operator+=(const LabeledBlockedTensorDistributive &rhs);
    void operator-=(const LabeledBlockedTensorDistributive &rhs);

    void operator=(const LabeledBlockedTensorProduct &rhs);
    void operator+=(const LabeledBlockedTensorProduct &rhs);
    void operator-=(const LabeledBlockedTensorProduct &rhs);

    void operator=(const LabeledBlockedTensorAddition &rhs);
    void operator+=(const LabeledBlockedTensorAddition &rhs);
    void operator-=(const LabeledBlockedTensorAddition &rhs);

    void operator*=(double scale);
    void operator/=(double scale);

    size_t numdim() const { return indices_.size(); }

    // negation
    LabeledBlockedTensor operator-() const
    {
        return LabeledBlockedTensor(BT_, indices_, -factor_);
    }

  private:
    void set(const LabeledBlockedTensor &to);

    std::vector<std::vector<size_t>> label_to_block_keys() const
    {
        return BT_.label_to_block_keys(indices_);
    }

    void contract(const LabeledBlockedTensorProduct &rhs, bool zero_result,
                  bool add);
    void add(const LabeledBlockedTensor &rhs, double alpha, double beta);

    BlockedTensor BT_;
    std::vector<std::string> indices_;
    double factor_;
};

inline LabeledBlockedTensor operator*(double factor,
                                      const LabeledBlockedTensor &ti)
{
    return LabeledBlockedTensor(ti.BT(), ti.indices(), factor * ti.factor());
};

class LabeledBlockedTensorProduct
{

  public:
    LabeledBlockedTensorProduct(const LabeledBlockedTensor &A,
                                const LabeledBlockedTensor &B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    size_t size() const { return tensors_.size(); }

    std::string str() const;

    const LabeledBlockedTensor &operator[](size_t i) const
    {
        return tensors_[i];
    }

    LabeledBlockedTensorProduct &operator*(const LabeledBlockedTensor &other)
    {
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
    LabeledBlockedTensorAddition(const LabeledBlockedTensor &A,
                                 const LabeledBlockedTensor &B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    size_t size() const { return tensors_.size(); }

    const LabeledBlockedTensor &operator[](size_t i) const
    {
        return tensors_[i];
    }

    std::vector<LabeledBlockedTensor>::iterator begin()
    {
        return tensors_.begin();
    }
    std::vector<LabeledBlockedTensor>::const_iterator begin() const
    {
        return tensors_.begin();
    }

    std::vector<LabeledBlockedTensor>::iterator end() { return tensors_.end(); }
    std::vector<LabeledBlockedTensor>::const_iterator end() const
    {
        return tensors_.end();
    }

    LabeledBlockedTensorAddition &operator+(const LabeledBlockedTensor &other)
    {
        tensors_.push_back(other);
        return *this;
    }

    LabeledBlockedTensorAddition &operator-(const LabeledBlockedTensor &other)
    {
        tensors_.push_back(-other);
        return *this;
    }

    LabeledBlockedTensorDistributive
    operator*(const LabeledBlockedTensor &other);

    LabeledBlockedTensorAddition &operator*(double scalar);

    // negation
    LabeledBlockedTensorAddition &operator-();

  private:
    // This handles cases like T("ijab")
    std::vector<LabeledBlockedTensor> tensors_;
};

inline LabeledBlockedTensorAddition
operator*(double factor, const LabeledBlockedTensorAddition &ti)
{
    LabeledBlockedTensorAddition ti2 = ti;
    return ti2 * factor;
}

// Is responsible for expressions like D * (J - K) --> D*J - D*K
class LabeledBlockedTensorDistributive
{
  public:
    LabeledBlockedTensorDistributive(const LabeledBlockedTensor &A,
                                     const LabeledBlockedTensorAddition &B)
        : A_(A), B_(B)
    {
    }

    const LabeledBlockedTensor &A() const { return A_; }
    const LabeledBlockedTensorAddition &B() const { return B_; }

    // conversion operator
    operator double() const;

  private:
    const LabeledBlockedTensor &A_;
    const LabeledBlockedTensorAddition &B_;
};

/// Take a string like "oovv" and generates the strings "oovv","oOvV","OOVV"
std::vector<std::string> spin_cases(const std::vector<std::string> &str);
}

#endif
