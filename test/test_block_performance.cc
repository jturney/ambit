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
    int repeats = 1;
    const size_t nc = 8, na = 4, nvDZ = 60, nvTZ = 160, mDZ = 280, mTZ = 444;
    size_t nv = nvDZ, m = mDZ;

    ambit::print("nc %zu; na %zu; nv %zu; m %zu, repeats %d\n", nc, na, nv, m, repeats);

    // define space labels
    std::string acore_label_ = "c";
    std::string aactv_label_ = "a";
    std::string avirt_label_ = "v";
    std::string bcore_label_ = "C";
    std::string bactv_label_ = "A";
    std::string bvirt_label_ = "V";

    std::vector<size_t> core_mos_(nc);
    for (size_t i = 0; i < nc; ++i) {
        core_mos_[i] = i;
    }
    std::vector<size_t> actv_mos_(na);
    for (size_t i = 0; i < na; ++i) {
        actv_mos_[i] = i;
    }
    std::vector<size_t> virt_mos_(nv);
    for (size_t i = 0; i < nv; ++i) {
        virt_mos_[i] = i;
    }
    // add Ambit index labels
    BlockedTensor::add_mo_space(acore_label_, "mn", core_mos_, AlphaSpin);
    BlockedTensor::add_mo_space(bcore_label_, "MN", core_mos_, BetaSpin);
    BlockedTensor::add_mo_space(aactv_label_, "uvwxyz123", actv_mos_, AlphaSpin);
    BlockedTensor::add_mo_space(bactv_label_, "UVWXYZ!@#", actv_mos_, BetaSpin);
    BlockedTensor::add_mo_space(avirt_label_, "ef", virt_mos_, AlphaSpin);
    BlockedTensor::add_mo_space(bvirt_label_, "EF", virt_mos_, BetaSpin);

    // define composite spaces
    BlockedTensor::add_composite_mo_space("h", "ijkl", {acore_label_, aactv_label_});
    BlockedTensor::add_composite_mo_space("H", "IJKL", {bcore_label_, bactv_label_});
    BlockedTensor::add_composite_mo_space("p", "abcd", {aactv_label_, avirt_label_});
    BlockedTensor::add_composite_mo_space("P", "ABCD", {bactv_label_, bvirt_label_});
    BlockedTensor::add_composite_mo_space("g", "pqrsto456", {acore_label_, aactv_label_, avirt_label_});
    BlockedTensor::add_composite_mo_space("G", "PQRSTO789", {bcore_label_, bactv_label_, bvirt_label_});

    // if DF/CD
    std::string aux_label_ = "L";
    std::vector<size_t> aux_mos_(m);
    for (size_t i = 0; i < m; ++i) {
        aux_mos_[i] = i;
    }
    BlockedTensor::add_mo_space(aux_label_, "g", aux_mos_, NoSpin);

    {
        BlockedTensor B = BlockedTensor::build(CoreTensor, "B 3-idx", {"Lgg", "LGG"});
        BlockedTensor C2 = BlockedTensor::build(CoreTensor, "C2", spin_cases({"gggg"}));
        BlockedTensor T2 = BlockedTensor::build(CoreTensor, "T2 Amplitudes", spin_cases({"hhpp"}));
        timing("1. C2[ijrs] += B[gar] * B[gbs] * T2[ijab];", repeats, [&]
               {
                   C2["ijrs"] += B["gar"] * B["gbs"] * T2["ijab"];
               });
    }

    {
        BlockedTensor B = BlockedTensor::build(CoreTensor, "B 3-idx", {"Lgg", "LGG"});
        BlockedTensor C2 = BlockedTensor::build(CoreTensor, "C2", spin_cases({"gggg"}));
        BlockedTensor T2 = BlockedTensor::build(CoreTensor, "T2 Amplitudes", spin_cases({"hhpp"}));
        timing("2. C2[ijrs] += batched(r, B[gar] * B[gbs] * T2[ijab]);", repeats, [&]
               {
                   C2["ijrs"] += batched("r", B["gar"] * B["gbs"] * T2["ijab"]);
               });
    }

    {
        BlockedTensor V = BlockedTensor::build(CoreTensor, "V", spin_cases({"gggg"}));
        BlockedTensor Vtrans = BlockedTensor::build(CoreTensor, "Vtrans", spin_cases({"gggg"}));
        BlockedTensor U1 = BlockedTensor::build(CoreTensor, "U", spin_cases({"gg"}));
        timing("3. Vtrans[pqrs] = U1[pt] * U1[qo] * V_[to45] * U1[r4] * U1[s5];", repeats, [&]
               {
                   Vtrans["pqrs"] = U1["pt"] * U1["qo"] * V["to45"] * U1["r4"] * U1["s5"];
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
