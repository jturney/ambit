#if !defined(TENSOR_TENSORIMPL_H)
#define TENSOR_TENSORIMPL_H

#include <string>
#include <vector>

namespace tensor {

class Tensor::TensorImpl
{
public:
    virtual TensorType type() const;
    std::string name() const { return name_; }
    virtual std::vector<size_t>& dims() const;
    virtual size_t rank() const;


private:
    std::string name_;
};

}

#endif

