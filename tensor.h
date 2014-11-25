
namespace tensor {

class TensorImpl;
class LabeledTensor;

enum TensorType { Core, Disk, Distributed, Agnostic }; 
enum EigenvalueOrder { Ascending, Descending };

class Tensor {
    
public:

    // => Constructors <= //

    static Tensor build(TensorType type, const std::string& name, const std::vector<size_t>& dims);

    static Tensor build(TensorType type, const Tensor& other);

    // => Reflectors <= //

    TensorType type() const { return tensor_->type(); }
    std::string name() const { return tensor_->name(); }
    const std::vector<size_t>& dims() const { return tensor_->dims(); }
    size_t rank() const { return tensor_->dims().size(); }
    size_t numel() const;

    /**
     * Print some tensor information to fh
     * If level = 0, just print name and dimensions
     * If level = 1, print the entire tensor
     **/ 
    void print(FILE* fh, int level = 0) const;

    // => Labelers <= // 

    LabeledTensor<Tensor> operator()(const std::string& indices);
    LabeledTensor<Tensor> operator[](const std::string& indices);

    // => Setters/Getters <= //

    void set_data(double* data, const std::vector<std::pair<size_t, size_t> >& ranges = std::vector<std::pair<size_t, size_t> >());
    void get_data(double* data, const std::vector<std::pair<size_t, size_t> >& ranges = std::vector<std::pair<size_t, size_t> >());

    static double* get_block(const std::vector<std::pair<size_t, size_t> >& ranges = std::vector<std::pair<size_t, size_t> >());
    static void free_block(double* data);

    // => Slicers <= //

    static Tensor slice(const Tensor& tensor, const std::vector<std::pair<size_t, size_t> >& ranges);
    static Tensor cat(const std::vector<Tensor>, int dim);

    // => Simple Single Tensor Operations <= // 

    Tensor& zero();
    Tensor& scale(double a);
    double norm(double power = 2.0) const;

    // => Simple Double Tensor Operations <= //

    Tensor& scale_and_add(double a, const Tensor& x);
    Tensor& pointwise_multiplication(const Tensor& x);
    Tensor& pointwise_division(const Tensor& x);
    double dot(const Tensor& x);

    // => Order-2 Operations <= //

    std::map<std::string, Tensor> syev(EigenValueOrder order);
    std::map<std::string, Tensor> geev(EigenValueOrder order);
    std::map<std::string, Tensor> svd();     

    Tensor cholesky();
    std::map<std::string, Tensor> lu();
    std::map<std::string, Tensor> qr();

    Tensor cholesky_inverse();
    Tensor inverse();
    Tensor power(double power, double condition = 1.0E-12);

    Tensor& givens(int dim, int i, int j, double s, double c);
    
private:

    boost::shared_ptr<TensorImpl> tensor_;

protected:

    Tensor(boost::shared_ptr<TensorImpl> tensor);

  

};

class LabeledTensor {

public:
    LabeledTensor(Tensor& T, const std::vector<std::string>& indices, double factor = 1.0) :
        T_(T), indices_(indices), factor_(factor)
    
    double factor() const { return factor_; }
    const std::vector<std::string>& indices() const { return indices_; }
    Tensor& T() const { return T_; }

    LabeledTensorProduct operator*(LabeledTensor& rhs);
        
    void operator=(LabeledTensor& rhs);
    void operator+=(LabeledTensor& rhs);
    void operator-=(LabeledTensor& rhs);

    void operator=(LabeledTensorProduct& rhs);
    void operator+=(LabeledTensorProduct& rhs);
    void operator-=(LabeledTensorProduct& rhs);

    void operator*=(double scale);
    

private:
    Tensor& T_;
    std::vector<std::string> indices_;
    double factor_;

};

inline LabeledTensor operator*(double factor, LabeledTensor& ti) {
    return LabeledTensor(ti.T(), ti.indices(), factor*ti.factor());
};

class LabeledTensorProduct {

public:
    LabeledTensorProduct(LabeledTensor& A, LabeledTensor& B) :
        A_(A), B_(B)

    LabeledTensor& A() { const return A_; }
    LabeledTensor& B() { const return B_; }

private:
    LabeledTensor& A_;
    LabeledTensor& B_;

}

}
