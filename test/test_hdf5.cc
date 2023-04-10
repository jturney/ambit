/*
 * @BEGIN LICENSE
 *
 * ambit: C++ library for the implementation of tensor product calculations
 *        through a clean, concise user interface.
 *
 * Copyright (c) 2014-2017 Ambit developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of ambit.
 *
 * Ambit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Ambit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <stdexcept>

#include <ambit/print.h>
#include <ambit/tensor.h>
#include <ambit/settings.h>
#include <ambit/io/hdf5/file.h>
#include <ambit/io/hdf5/group.h>
#include <ambit/io/hdf5/dataspace.h>
#include <ambit/io/hdf5/dataset.h>

#include <ambit/timer.h>

using namespace ambit;

TensorType tensor_type = CoreTensor;

Tensor build(const std::string& name, const Dimension& dims)
{
    return Tensor::build(tensor_type, name, dims);
}

void initialize_random(Tensor& tensor)
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    std::vector<double>& vec = tensor.data();
    for (size_t i = 0, ij = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j, ++ij) {
            double randnum = double(std::rand()) / double(RAND_MAX);
            vec[ij] = randnum;
        }
    }
}

void test_hdf5()
{
    using namespace ambit::io::hdf5;

    Tensor testTensor = build("Test Tensor", {7, 7});
    initialize_random(testTensor);

    File test("test.h5", kOpenModeCreateNew, kDeleteModeKeepOnClose);
    File test1("delete.h5", kOpenModeCreateNew, kDeleteModeDeleteOnClose);

    Group three = test.group("1").group("2").group("3");
    Dataspace space(testTensor);
    Dataset<double> set(three, testTensor.name(), space);
    set.write(testTensor);

    Dataspace dims({2});
    hid_t attribute_id = H5Acreate2(set.id(), "Dimensions", detail::ctype<size_t>::hid(),
                                   dims.id(), H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute_id, detail::ctype<size_t>::hid(), testTensor.dims().data());
    H5Aclose(attribute_id);

    Dataset<double>::write(test, testTensor);
    testTensor.set_name("test2");
    write(test, testTensor);

    // Save to the HDF5 file to test against Mathematica.
    auto result = testTensor.gesvd();
    write(test, result["U"]);
    write(test, result["V"]);
    write(test, result["Sigma"]);
}

int main(int argc, char *argv[])
{
    srand(time(nullptr));
    ambit::settings::timers = true;
    ambit::initialize(argc, argv);

    if (argc > 1) {
        if (settings::distributed_capable && strcmp(argv[1], "cyclops") == 0) {
            tensor_type = DistributedTensor;
            ambit::print("  *** Testing distributed tensors. ***\n");
            ambit::print("      Running in %d processes.\n",
                         ambit::settings::nprocess);
        }
        else {
            ambit::print("  *** Unknown parameter given ***\n");
            ambit::print("  *** Testing core tensors.   ***\n");
        }
    }

    test_hdf5();

    ambit::finalize();
    return EXIT_SUCCESS;
}
