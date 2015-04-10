#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <assert.h>

#include <ambit/print.h>
#include <ambit/tensor.h>
#include <ambit/timer.h>

#include <ostream>
#include <iostream>
#include <iomanip>

using namespace ambit;

TensorType tensor_type = kCore;

void progress_bar(const std::string &label, int len, double percent)
{
    ambit::print("\x1b[2K"); // Erase the entire current line.
    ambit::print("\r"); // Move to the beginning of the current line.
    std::string progress;
    for (int i = 0; i < len; ++i) {
        if (i < static_cast<int>(len * percent)) {
            progress += "=";
        } else {
            progress += " ";
        }
    }
    ambit::print("%-35s: [%s] %d%%", label.c_str(), progress.c_str(), static_cast<int>(100 * percent));
    fflush(stdout);
}

Tensor build(const std::string &name, const Dimension &dims)
{
    return Tensor::build(tensor_type, name, dims);
}

void timing(const std::string &label, int repeats, const std::function<void()> &test)
{
    ambit::timer::timer_push(label);
    progress_bar(label, 50, 0);
    for (int i = 0; i < repeats; i++) {
        test();
        progress_bar(label, 50, double(i + 1) / double(repeats));
    }
    ambit::timer::timer_pop();
    ambit::print("\n");
}

void test_performance()
{
    size_t nv = 100;
    size_t no = 25;
    int repeats = 10;

    ambit::print("no %d; nv %d, repeats %d\n", no, nv, repeats);

    Tensor T1 = build("T1", {no, nv});
//    Tensor T1t = build("T1t", {nv, no});

    {
        Tensor Wbaef = build("Wbaef", {nv, nv, nv, nv});
        Tensor Giabc = build("Giabc", {no, nv, nv, nv});
        timing("1. Wbaef = T1mb * Gmaef", repeats, [&] {
            Wbaef("b,a,e,f") = T1("m,b") * Giabc("m,a,e,f");
        });
    }

    {
        Tensor Gaibc = build("Gaibc", {nv, no, nv, nv});
        Tensor Wabef = build("Wabef", {nv, nv, nv, nv});
        timing("2. Wabef = T1mb * Gamef", repeats, [&] {
            Wabef("a,b,e,f") = T1("m,b") * Gaibc("a,m,e,f");
        });
    }

    {
        Tensor Gabic = build("Gabic", {nv, nv, no, nv});
        Tensor Waebf = build("Waebf", {nv, nv, nv, nv});
        timing("3. Waebf = T1mb * Gaemf", repeats, [&] {
            Waebf("a,e,b,f") = T1("m,b") * Gabic("a,e,m,f");
        });
    }

    {
        Tensor Waefb = build("Waefb", {nv, nv, nv, nv});
        Tensor Gabci = build("Gabci", {nv, nv, nv, no});
        timing("4. Waefb = T1mb * Gaefm", repeats, [&] {
            Waefb("a,e,f,b") = T1("m,b") * Gabci("a,e,f,m");
        });
    }

//    {
//        Tensor Wefab = build("Wefab", {nv, nv, nv, nv});
//        Tensor Gabci = build("Gabci", {nv, nv, nv, no});
//        timing("3. Wabef = Wefab = T1tbm * Gefam", repeats, [&] {
//            Wefab("e,f,a,b") = T1t("b,m") * Gabci("e,f,a,m");
////        Wabef("a,b,e,f") -= Wefab("e,f,a,b");
//        });
//    }

}

int main(int argc, char *argv[])
{
    srand(time(nullptr));
    ambit::settings::timers = true;
    ambit::initialize(argc, argv);

    if (argc > 1) {
        if (settings::distributed_capable && strcmp(argv[1], "cyclops") == 0) {
            tensor_type = kDistributed;
            ambit::print("  *** Testing distributed tensors. ***\n");
            ambit::print("      Running in %d processes.\n", ambit::settings::nprocess);
        }
        else {
            ambit::print("  *** Unknown parameter given ***\n");
            ambit::print("  *** Testing core tensors.   ***\n");
        }
    }

    test_performance();

    ambit::finalize();
    return EXIT_SUCCESS;
}
