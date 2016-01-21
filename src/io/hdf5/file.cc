//
// Created by Justin Turney on 1/8/16.
//

#include <ambit/io/hdf5/file.h>
#include <ambit/print.h>

namespace ambit {

namespace io {

namespace hdf5 {

File::File(const string& filename, OpenMode om, DeleteMode dm)
        : Location(), filename_(filename)
{
    open(filename, om, dm);
}

File::~File()
{
    close();
}

void File::open(const string& filename, OpenMode om, DeleteMode dm)
{
    delete_mode_ = dm;

    if (om == kOpenModeCreateNew)
        id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    else
        id_ = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
}

void File::close()
{
    if (id_ != -1) {
        H5Fclose(id_);

        if (delete_mode_ == kDeleteModeDeleteOnClose)
            remove(filename_.c_str());

        id_ = -1;
    }
}

} // namespace hdf6

} // namespace io

} // namespace ambit
