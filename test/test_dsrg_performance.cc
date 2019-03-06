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
#include <ambit/timer.h>
#include <ambit/blocked_tensor.h>

#include <ostream>
#include <iostream>
#include <iomanip>

using namespace ambit;

TensorType tensor_type = CoreTensor;

void progress_bar(const std::string &label, int len, double percent)
{
    ambit::print("\x1b[2K"); // Erase the entire current line.
    ambit::print("\r");      // Move to the beginning of the current line.
    std::string progress;
    for (int i = 0; i < len; ++i)
    {
        if (i < static_cast<int>(len * percent))
        {
            progress += "=";
        }
        else
        {
            progress += " ";
        }
    }
    ambit::print("%-35s: [%s] %d%%", label.c_str(), progress.c_str(),
                 static_cast<int>(100 * percent));
    fflush(stdout);
}

Tensor build(const std::string &name, const Dimension &dims)
{
    return Tensor::build(tensor_type, name, dims);
}

void timing(const std::string &label, int repeats,
            const std::function<void()> &test)
{
    ambit::timer::timer_push(label);
    progress_bar(label, 50, 0);
    for (int i = 0; i < repeats; i++)
    {
        test();
        progress_bar(label, 50, double(i + 1) / double(repeats));
    }
    ambit::timer::timer_pop();
    ambit::print("\n");
}

void test_performance()
{
//    size_t nv = 100;
//    size_t no = 25;
    int repeats = 1;
    const size_t nc = 8, na = 4, nvDZ = 60, nvTZ = 160, mDZ = 280, mTZ = 444;
    size_t nv = nvDZ, m = mDZ;

//    ambit::print("no %d; nv %d, repeats %d\n", no, nv, repeats);
    ambit::print("nc %zu; na %zu; nv %zu; m %zu, repeats %d\n", nc, na, nv, m, repeats);

//    Tensor H2, T2, C2, B, Bp, Bb, w, C2p, C2b;

    {
        Tensor B = build("B[Qar/Qbs]", {m, nv, nv});
        Tensor w = build("w[arbs]", {nv, nv, nv, nv});
        timing("1. w(arbs) = B(Qar) * B(Qbs)", repeats, [&]
               {
                   w("arbs") = B("Qar") * B("Qbs");
               });
    }
    {
        Tensor H2 = build("H2[abrs]", {nv, nv, nv, nv});
        Tensor B = build("B[Qar/Qbs]", {m, nv, nv});
        timing("2. H2(abrs) = B(Qar) * B(Qbs)", repeats, [&]
               {
                   H2("abrs") = B("Qar") * B("Qbs");
               });
    }
    {
        Tensor H2 = build("H2[abrs]", {nv, nv, nv, nv});
        Tensor C2 = build("C2[ijrs]", {nc, nc, nv, nv});
        Tensor T2 = build("T2[ijab]", {nc, nc, nv, nv});
        timing("3. C2(ijrs) += H2(abrs) * T2(ijab)", repeats, [&]
               {
                   C2("ijrs") += H2("abrs") * T2("ijab");
               });
    }
    {
        Tensor B = build("B[Qar/Qbs]", {m, nv, nv});
        Tensor C2 = build("C2[ijrs]", {nc, nc, nv, nv});
        Tensor T2 = build("T2[ijab]", {nc, nc, nv, nv});
        timing("4. C2(ijrs) += B(Qar) * B(Qbs) * T2(ijab)", repeats, [&]
               {
                   C2("ijrs") += B("Qar") * B("Qbs") * T2("ijab");
               });
    }
    {
        Tensor B = build("B[Qar/Qbs]", {m, nv, nv});
        Tensor Bb = build("Bb[Qa]", {m, nv});
        Tensor Bc = build("Bc[Qa]", {m, nv});
        Tensor Cb = build("Cb[ijs]", {nc, nc, nv});
        Tensor Cc = build("Cc[ijs]", {nc, nc, nv});
        Tensor C2 = build("C2[ijrs]", {nc, nc, nv, nv});
        Tensor T2 = build("T2[ijab]", {nc, nc, nv, nv});
        timing("5. C2(ijrs) += detailbatched(r, B(Qar) * B(Qbs) * T2(ijab))", repeats, [&]
               {
                   ambit::timer::timer_push("Build Bp[rQa]");
                   Tensor Bp = build("Bp[rQa]", {nv, m, nv});
                   ambit::timer::timer_pop();
                   ambit::timer::timer_push("Build Cp[rijs]");
                   Tensor Cp = build("Cp[rijs]", {nv, nc, nc, nv});
                   ambit::timer::timer_pop();
                   Bp("rQa") = B("Qar");
                   Cp("rijs") = C2("ijrs");
                   for (size_t i = 0; i < nv; ++i) {
                       Bb("Qa") = Bc("Qa");
                       ambit::timer::timer_push("Build wb[abs]");
                       Tensor wb = build("wb[abs]", {nv, nv, nv});
                       ambit::timer::timer_pop();
                       wb("abs") = Bb("Qa") * B("Qbs");
                       Cb("ijs") = wb("abs") * T2("ijab");
                       Cc("ijs") = Cb("ijs");
                   }
                   C2("ijrs") = Cp("rijs");
               });
    }
    {
        Tensor B = build("B[Qar/Qbs]", {m, nv, nv});
        Tensor C2 = build("C2[ijrs]", {nc, nc, nv, nv});
        Tensor T2 = build("T2[ijab]", {nc, nc, nv, nv});
        timing("6. C2(ijrs) += batched(r, B(Qar) * B(Qbs) * T2(ijab))", repeats, [&]
               {
                   C2("ijrs") += batched("r", B("Qar") * B("Qbs") * T2("ijab"));
               });
    }
}

int main(int argc, char *argv[])
{
    srand(time(nullptr));
    ambit::settings::timers = true;
    ambit::initialize(argc, argv);

    if (argc > 1)
    {
        if (settings::distributed_capable && strcmp(argv[1], "cyclops") == 0)
        {
            tensor_type = DistributedTensor;
            ambit::print("  *** Testing distributed tensors. ***\n");
            ambit::print("      Running in %d processes.\n",
                         ambit::settings::nprocess);
        }
        else
        {
            ambit::print("  *** Unknown parameter given ***\n");
            ambit::print("  *** Testing core tensors.   ***\n");
        }
    }

    test_performance();

    ambit::finalize();
    return EXIT_SUCCESS;
}
