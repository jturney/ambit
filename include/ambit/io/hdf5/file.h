//
// Created by Justin Turney on 1/8/16.
//

#ifndef AMBIT_FILE_H
#define AMBIT_FILE_H

#include <ambit/common_types.h>
#include <ambit/io/hdf5/group.h>
#include <ambit/io/hdf5/location.h>

namespace ambit {

namespace io {

namespace hdf5 {

enum OpenMode
{
    kOpenModeCreateNew,
    kOpenModeOpenExisting
};

enum DeleteMode
{
    kDeleteModeKeepOnClose,
    kDeleteModeDeleteOnClose
};

struct File
        : public Location
{
    File() = default;
    File(const string& filename,
         OpenMode om,
         DeleteMode dm = kDeleteModeKeepOnClose);

    virtual ~File();

    void open(const string& filename,
              OpenMode om,
              DeleteMode dm = kDeleteModeKeepOnClose);

    void close();

private:
    string filename_;
    DeleteMode delete_mode_;
};

} // namespace hdf5

} // namespace io

} // namespace ambit
#endif //AMBIT_FILE_H
