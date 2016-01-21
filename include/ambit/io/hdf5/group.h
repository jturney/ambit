//
// Created by Justin Turney on 1/8/16.
//

#ifndef AMBIT_GROUP_H
#define AMBIT_GROUP_H

#include <ambit/common_types.h>
#include <ambit/io/hdf5/location.h>

namespace ambit {

namespace io {

namespace hdf5 {

struct Location;

struct Group : public Location
{
    Group(const Location& loc, const string& name);
    virtual ~Group();

    size_t size() const;

private:

    void create(const Location& loc, const string& name);
    void open(const Location& loc, const string& name);
    void close();

};

} // namespace hdf5

} // namespace io

} // namespace ambit
#endif //AMBIT_GROUP_H
