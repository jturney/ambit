//
// Created by Justin Turney on 1/9/16.
//

#include <ambit/common_types.h>
#include <ambit/io/hdf5/group.h>

namespace ambit {

namespace io {

namespace hdf5 {

Group::Group(const Location& loc, const string& name)
{
    if (loc.has_link(name))
        open(loc, name);
    else
        create(loc, name);
}

Group::~Group()
{
    close();
}

void Group::create(const Location& loc, const string& name)
{
    close();
    id_ = H5Gcreate2(loc.id(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(id_  >= 0);
}

void Group::open(const Location& loc, const string& name)
{
    close();
    id_ = H5Gopen2(loc.id(), name.c_str(), H5P_DEFAULT);
    assert(id_ >= 0);
}

void Group::close()
{
    if (id_ >= 0)
        H5Gclose(id_);
    id_ = -1;
}

size_t Group::size() const
{
    hsize_t size;
    H5Gget_num_objs(id(), &size);
    return static_cast<size_t>(size);
}

} // namespace hdf5

} // namespace io

} // namespace ambit
