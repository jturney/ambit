//
// Created by Justin Turney on 1/8/16.
//

#ifndef AMBIT_LOCATION_H
#define AMBIT_LOCATION_H

#include <ambit/common_types.h>
#include <hdf5.h>

namespace ambit {

namespace io {

namespace hdf5 {

struct Group;

struct Location
{
    Location(hid_t id = -1);
    virtual ~Location();

    hid_t id() const noexcept;

    bool has_link(const string& name) const;

    Group group(const string& name) const;

//    hsize_t

protected:
    hid_t id_;
};

} // namespace hdf5

} // namespace io

} // namespace ambit

#endif //AMBIT_LOCATION_H
