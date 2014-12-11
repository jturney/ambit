#include "tensorimpl.h"
#include "core.h"

namespace tensor {

CoreTensorImpl::CoreTensorImpl(const std::string& name, const Dimension& dims)
        : TensorImpl(Core, name, dims)
{
    data_ = new double[numel()];
    memset(data_,'\0', sizeof(double)*numel());
}
CoreTensorImpl::~CoreTensorImpl()
{
    if (data_) delete[] data_;
}
void CoreTensorImpl::set_data(double* data, const IndexRange& range)
{
    if (range.size() == 0) {
        memcpy(data_,data,sizeof(double)*numel());
        return;
    }
    // TODO
}
void CoreTensorImpl::get_data(double* data, const IndexRange& range) const
{
    if (range.size() == 0) {
        memcpy(data,data_,sizeof(double)*numel());
        return;
    }
    // TODO
}
void CoreTensorImpl::zero()
{
    memset(data_,'\0', sizeof(double)*numel());
}


}
