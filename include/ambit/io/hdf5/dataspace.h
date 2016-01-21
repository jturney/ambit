//
// Created by Justin Turney on 1/11/16.
//

#ifndef AMBIT_DATASPACE_H_H
#define AMBIT_DATASPACE_H_H

#include <ambit/common_types.h>
#include <ambit/tensor.h>
#include <hdf5.h>

namespace ambit {

namespace io {

namespace hdf5 {

struct Dataspace
{

    Dataspace(const Tensor& tensor);
    Dataspace(const Dimension& current_dims);
    Dataspace(const Dimension& current_dims, const Dimension& maximum_dims);

    virtual ~Dataspace();

    hid_t id() const
    {
        return dataspace_id_;
    }

private:
    hid_t dataspace_id_;
};

} // namespace hdf5

} // namespace io

} // namespace ambit
#endif //AMBIT_DATASPACE_H_H
