//
// Created by Justin Turney on 1/20/16.
//

#include <ambit/io/hdf5/attribute.h>

namespace ambit {

namespace io {

namespace hdf5 {

Attribute::Attribute(Location const& location, const string& name)
        : id_(-1), location_(location), name_(name)
{ }

Attribute::~Attribute()
{
    if (id_ != -1) {
        H5Aclose(id_);
        id_ = -1;
    }
}

bool Attribute::exists() const
{
    return H5Aexists(location_.id(), name_.c_str()) > 0;
}

} // namespace hdf5

} // namespace io

} // namespace ambit
