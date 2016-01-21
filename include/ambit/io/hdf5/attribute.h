//
// Created by Justin Turney on 1/20/16.
//

#ifndef AMBIT_ATTRIBUTE_H
#define AMBIT_ATTRIBUTE_H

#include <ambit/common_types.h>
#include <ambit/io/hdf5/location.h>
#include <hdf5.h>

namespace ambit {

namespace io {

namespace hdf5 {

struct Attribute
{

    Attribute(const Location& location, const string& name);
    virtual ~Attribute();

    const hid_t& id() const { return id_; }

    bool exists() const;

private:
    hid_t id_;

    const Location& location_;
    const string& name_;
};

} // namespace hdf5

} // namespace io

} // namespace ambit
#endif //AMBIT_ATTRIBUTE_H
