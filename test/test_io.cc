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
 * You should have received a copy of the GNU Lesser General Public License
 * along with ambit; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include <ambit/blocked_tensor.h>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"

#define MAXTWO 10
#define MAXFOUR 10

/// Scheme to categorize expected vs. actual op behavior
enum TestResult
{
    kPass,
    kFail,
    kException
};

using namespace ambit;

/// Expected relative accuracy for numerical exactness
const double epsilon = 1.0E-14;
const double zero = 1.0E-14;

TensorType tensor_type = CoreTensor;

Tensor build(const std::string &name, const Dimension &dims)
{
    return Tensor::build(tensor_type, name, dims);
}

void initialize_random(Tensor &tensor)
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    std::vector<double> &vec = tensor.data();
    for (size_t i = 0, ijk = 0; i < n0; ++i)
    {
        for (size_t j = 0; j < n1; ++j)
        {
            for (size_t k = 0; k < n2; ++k, ++ijk)
            {
                double randnum = double(std::rand()) / double(RAND_MAX);
                vec[ijk] = randnum;
            }
        }
    }
}

bool test_tensor_io_1()
{
    // create a random tensor
    Tensor testTensor = build("Test Tensor", {3, 7, 7});
    initialize_random(testTensor);

    // save the tensor
    save(testTensor, "test.ten");

    // load the data from disk
    Tensor testTensor2 = build("Test Tensor 2", {7, 7, 3});
    load(testTensor2, "test.ten");

    // compare the data to the original tensors
    testTensor2("abc") += -testTensor("abc");
    return testTensor2.norm();
}

bool test_tensor_io_2()
{
    // create a random tensor
    Tensor testTensor = build("Test Tensor", {3, 7, 7});
    initialize_random(testTensor);

    // save the tensor
    save(testTensor, "test.ten");

    // load the data from disk
    Tensor testTensor2;
    load(testTensor2, "test.ten");

    // compare the data to the original tensors
    testTensor2("abc") += -testTensor("abc");
    return testTensor2.norm();
}

bool test_tensor_io_3()
{
    // create a random tensor
    Tensor testTensor = build("Test Tensor", {3, 7, 7});
    initialize_random(testTensor);

    // save the tensor
    save(testTensor, "test.ten");

    // load the data from disk
    Tensor testTensor2 = load_tensor("test.ten");

    // compare the data to the original tensors
    testTensor2("abc") += -testTensor("abc");
    return testTensor2.norm();
}

bool test_tensor_io_blocked_1()
{
    // register the orbital spaces with the class
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("O", "i,j,k,l", {0, 1, 2, 3, 4}, AlphaSpin);
    BlockedTensor::add_mo_space("V", "a,b,c,d", {7, 8, 9}, AlphaSpin);

    // instantiate BlockTensor objects
    BlockedTensor testTensor =
        BlockedTensor::build(CoreTensor, "T", {"OOO", "OOV"});

    // create a random tensor
    Tensor t_ooo = testTensor.block("OOO");
    Tensor t_oov = testTensor.block("OOV");
    initialize_random(t_ooo);
    initialize_random(t_oov);
    testTensor.set_block("OOO", t_ooo);
    testTensor.set_block("OOV", t_oov);

    // save the tensor
    save(testTensor, "block.ten");

    // load the data from disk
    BlockedTensor testTensor2 =
        BlockedTensor::build(CoreTensor, "T", {"OOO", "OOV"});
    load(testTensor2, "block.ten");

    // compare the data to the original tensors
    testTensor2("ijk") += -testTensor("ijk");
    testTensor2("ija") += -testTensor("ija");

    return testTensor2.norm();
}

bool test_tensor_io_blocked_2()
{
    // register the orbital spaces with the class
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("O", "i,j,k,l", {0, 1, 2, 3, 4}, AlphaSpin);
    BlockedTensor::add_mo_space("V", "a,b,c,d", {7, 8, 9}, AlphaSpin);

    // instantiate BlockTensor objects
    BlockedTensor testTensor =
        BlockedTensor::build(CoreTensor, "T", {"OOO", "OOV"});

    // create a random tensor
    Tensor t_ooo = testTensor.block("OOO");
    Tensor t_oov = testTensor.block("OOV");
    initialize_random(t_ooo);
    initialize_random(t_oov);
    testTensor.set_block("OOO", t_ooo);
    testTensor.set_block("OOV", t_oov);

    // save the tensor
    save(testTensor, "block.ten");

    // load the data from disk
    BlockedTensor testTensor2;
    load(testTensor2, "block.ten");

    // compare the data to the original tensors
    testTensor2("ijk") += -testTensor("ijk");
    testTensor2("ija") += -testTensor("ija");

    return testTensor2.norm();
}

bool test_tensor_io_blocked_3()
{
    // register the orbital spaces with the class
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("O", "i,j,k,l", {0, 1, 2, 3, 4}, AlphaSpin);
    BlockedTensor::add_mo_space("V", "a,b,c,d", {7, 8, 9}, AlphaSpin);

    // instantiate BlockTensor objects
    BlockedTensor testTensor =
        BlockedTensor::build(CoreTensor, "T", {"OOO", "OOV"});

    // create a random tensor
    Tensor t_ooo = testTensor.block("OOO");
    Tensor t_oov = testTensor.block("OOV");
    initialize_random(t_ooo);
    initialize_random(t_oov);
    testTensor.set_block("OOO", t_ooo);
    testTensor.set_block("OOV", t_oov);

    // save the tensor
    save(testTensor, "block.ten");

    // load the data from disk
    BlockedTensor testTensor2 = load_blocked_tensor("block.ten");

    // compare the data to the original tensors
    testTensor2("ijk") += -testTensor("ijk");
    testTensor2("ija") += -testTensor("ija");

    return testTensor2.norm();
}

int main(int argc, char *argv[])
{
    printf(ANSI_COLOR_RESET);
    srand(time(nullptr));
    ambit::initialize(argc, argv);

    printf("==> Tensor I/O Operations <==\n\n");

    auto test_functions = {
        //            Expectation,  test function,  User friendly description
        std::make_tuple(kPass, test_tensor_io_1,
                        "Save/load Tensor (on a pre-allocated tensor)"),
        std::make_tuple(kPass, test_tensor_io_2,
                        "Save/load Tensor (on an empty tensor)"),
        std::make_tuple(kPass, test_tensor_io_3,
                        "Save/load Tensor (return a tensor)"),
        std::make_tuple(kPass, test_tensor_io_blocked_2,
                        "Save/load BlockedTensor (on a pre-allocated tensor)"),
        std::make_tuple(kPass, test_tensor_io_blocked_2,
                        "Save/load BlockedTensor (on an empty tensor)"),
        std::make_tuple(kPass, test_tensor_io_blocked_3,
                        "Save/load BlockedTensor (return a tensor)"),
    };

    std::vector<std::tuple<std::string, TestResult, double>> results;

    printf(ANSI_COLOR_RESET);

    printf("\n %-60s %12s %s", "Description", "Max. error", "Result");
    printf("\n %s", std::string(83, '-').c_str());

    bool success = true;
    for (auto test_function : test_functions)
    {
        printf("\n %-60s", std::get<2>(test_function));
        double result = 0.0;
        TestResult tresult = kPass, report_result = kPass;
        std::string exception;
        try
        {
            result = std::get<1>(test_function)();

            // Did the test pass based on returned value?
            tresult = std::fabs(result) < epsilon ? kPass : kFail;
            // Was the tresult the expected result? If so color green else red.
            report_result =
                tresult == std::get<0>(test_function) ? kPass : kFail;
        }
        catch (std::exception &e)
        {
            // was an exception expected?
            tresult = kException;
            report_result =
                tresult == std::get<0>(test_function) ? kPass : kException;

            //            printf("\n  %s",e.what());
            if (report_result == kException)
            {
                exception = e.what();
            }
        }
        printf(" %7e", result);
        switch (report_result)
        {
        case kPass:
            printf(ANSI_COLOR_GREEN);
            break;
        case kFail:
            printf(ANSI_COLOR_RED);
            break;
        default:
            printf(ANSI_COLOR_YELLOW);
        }
        switch (tresult)
        {
        case kPass:
            printf(" Passed" ANSI_COLOR_RESET);
            break;
        case kFail:
            printf(" Failed" ANSI_COLOR_RESET);
            break;
        default:
            printf(" Exception" ANSI_COLOR_RESET);
        }

        if (report_result == kException)
            printf("\n    Unexpected: %s", exception.c_str());
        if (report_result != kPass)
            success = false;
    }
    printf("\n %s", std::string(83, '-').c_str());
    printf("\n Tests: %s\n", success ? "All passed" : "Some failed");

    ambit::finalize();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
