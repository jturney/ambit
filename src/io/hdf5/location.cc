//
// Created by Justin Turney on 1/8/16.
//

#include <ambit/io/hdf5/location.h>
#include <ambit/io/hdf5/group.h>

namespace ambit {

namespace io {

namespace hdf5 {

Location::Location(hid_t id)
        : id_(id)
{}

Location::~Location()
{}

hid_t Location::id() const noexcept
{
    return id_;
}

bool Location::has_link(string const& name) const
{
    htri_t res = H5Lexists(id_, name.c_str(), H5P_DEFAULT);
    assert(res >= 0);
    return static_cast<bool>(res);
}

Group Location::group(const string& name) const
{
    return Group(*this, name);
}

} // namespace hdf5

} // namespace io

} // namespace ambit
