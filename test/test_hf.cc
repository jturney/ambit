#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <assert.h>

#include <ambit/print.h>
#include <ambit/tensor.h>
#include <ambit/io/io.h>
#include <ambit/helpers/psi4/io.h>
#include <ambit/timer.h>

using namespace ambit;

TensorType tensor_type = kCore;

Tensor build(const std::string &name, const Dimension &dims)
{
    return Tensor::build(tensor_type, name, dims);
}

Tensor build_and_load(const std::string &file35, const std::string &toc,
                      const Dimension &AO)
{
    Tensor X = build(toc, AO);
    helpers::psi4::load_matrix(file35, toc, X);
    return X;
}

Tensor load_overlap(const std::string &file35, const Dimension &AO)
{
    return build_and_load(file35, "SO-basis Overlap Ints", AO);
}

Tensor load_1e_hamiltonian(const std::string &file35, const Dimension &AO)
{
    Tensor H = build("H", AO);

    Tensor T = build_and_load(file35, "SO-basis Kinetic Energy Ints", AO);
    Tensor V = build_and_load(file35, "SO-basis Potential Energy Ints", AO);

    H("p,q") = T("p,q") + V("p,q");

    return H;
}

Tensor load_2e(const Dimension &AO)
{
    Tensor g = build("g", AO);
    helpers::psi4::load_iwl("test.33", g);
    return g;
}

void hf()
{
    int nirrep, nso;
    double Enuc = 0.0, Eref = 0.0;

    {
        ambit::io::File file32("test.32", ambit::io::kOpenModeOpenExisting);

        file32.read("::Num. irreps", &nirrep, 1);
        print("nirrep = %d\n", nirrep);
        assert(nirrep == 1);

        file32.read("::Num. SO", &nso, 1);
        print("nso = %d\n", nso);

        file32.read("::Nuclear rep. energy", &Enuc, 1);
        file32.read("::SCF energy", &Eref, 1);
    }

    // Define dimension objects
    Dimension AO2 = {(size_t)nso, (size_t)nso};
    Dimension AO4 = {(size_t)nso, (size_t)nso, (size_t)nso, (size_t)nso};

    // Build tensors
    Tensor S = load_overlap("test.35", AO2);
    print("norm of S is %lf\n", S.norm());
    Tensor H = load_1e_hamiltonian("test.35", AO2);
    Tensor g = load_2e(AO4);
    print("norm of g is %lf\n", g.norm());

    Tensor Ft = build("Ft", AO2);
    Tensor Smhalf = S.power(-0.5);
    //    Smhalf.print(stdout, true);

    Ft("i,j") = Smhalf("mu,i") * Smhalf("nu,j") * H("mu,nu");
    //    Ft.print(stdout, true);
    auto Feigen = Ft.syev(kAscending);
    //    Feigen["eigenvectors"].print(stdout, true);

    Tensor C = build("C", AO2);
    C("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");
    //    C.print(stdout, true);

    Tensor Cdocc = build("C", {5, (size_t)nso});

    size_t ndocc = 5;
    IndexRange CtoCdocc = {{0, ndocc}, {0, (size_t)nso}};
    // Cdocc.slice(C, CtoCdocc, CtoCdocc);
    Cdocc(CtoCdocc) = C(CtoCdocc);

    // Form initial D
    Tensor D = build("D", AO2);
    D("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");

    Tensor F = build("F", AO2);

    // start SCF iteration
    bool converged = false;
    double Eelec = 0.0, Eold = 0.0;
    int iter = 1;
    do
    {
        ambit::timer::timer_push("HF iteration");

        F("mu,nu") = H("mu,nu");
        F("mu,nu") += D("rho,sigma") *
                      (2.0 * g("mu,nu,rho,sigma") - g("mu,rho,nu,sigma"));
        //        F.print(stdout, true);

        // Calculate energy
        Eelec = D("mu,nu") * (H("mu,nu") + F("mu,nu"));

        print("  @RHF iter %5d: %20.14lf\n", iter++, Enuc + Eelec);

        // Transform the Fock matrix
        Ft("i,j") = Smhalf("mu,i") * Smhalf("nu,j") * F("mu,nu");

        // Diagonalize Fock matrix
        Feigen = Ft.syev(kAscending);

        // Construct new SCF eigenvector matrix.
        C("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");

        // Form new density matrix
        Cdocc(CtoCdocc) = C(CtoCdocc);
        D("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");
        //        D.print(stdout, true);

        if (std::fabs(Eelec - Eold) < 1.0e-8)
            converged = true;
        Eold = Eelec;

        ambit::timer::timer_pop();

        if (iter > 15)
            break;
    } while (!converged);

    //    C.print(stdout, true);
    C.iterate([](const std::vector<size_t> & /*indices*/, double &value)
              {
                  value += 1.0;
              });
    //    C.print(stdout, true);

    //    C.citerate([](const std::vector<size_t>& indices, const double& value)
    //    {
    //        printf("rank %d: C[%lu, %lu] %lf\n", settings::rank, indices[0],
    //        indices[1], value);
    //    });

    //    g.citerate([](const std::vector<size_t>& indices, const double& value)
    //    {
    //        printf("g[%lu, %lu, %lu, %lu] %lf\n", indices[0], indices[1],
    //        indices[2], indices[3], value);
    //    });

    // the energy eigenvalues

    Tensor t_eigev = Tensor::build(kCore, "eigenvalues", {(size_t)nso});
    IndexRange all = {{0L, (size_t)nso}};
    t_eigev(all) = Feigen["eigenvalues"](all);
    std::vector<double> e_eigev = t_eigev.data();

    //    if (settings::rank == 0)
    t_eigev.print();

    // Construct denominators

    Tensor Dia = build("Dia", {5, 2});
    Dia.iterate([&](const std::vector<size_t> &indices, double &value)
                {
                    //        value =
                    //        1.0/(e_eigev[indices[0]]-e_eigev[indices[1]+ndocc]);
                    printf("indices %d %d: %lf %lf\n", indices[0], indices[1],
                           e_eigev[indices[0]], e_eigev[indices[1] + 5]);
                    value = 1.0 / (t_eigev.data()[indices[0]] -
                                   t_eigev.data()[indices[1] + 5]);
                });
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
            tensor_type = kDistributed;
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

    hf();

    ambit::finalize();
    return EXIT_SUCCESS;
}
