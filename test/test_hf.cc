#include <tensor/tensor.h>
#include <tensor/io/io.h>

#include <string>
#include <cmath>
#include <cstdlib>
#include <assert.h>

using namespace tensor;

Tensor build(const std::string& name, const Dimension& dims)
{
    return Tensor::build(kCore, name, dims);
}

Tensor build_and_load(io::File& file35, const std::string& toc, const Dimension& AO)
{
    Tensor X = build(toc, AO);
    X.zero();
    io::IWL::read_one(file35, toc, X);
    return X;
}

Tensor load_overlap(io::File& file35, const Dimension& AO)
{
    return build_and_load(file35, "SO-basis Overlap Ints", AO);
}

Tensor load_1e_hamiltonian(io::File& file35, const Dimension& AO)
{
    Tensor H = build("H", AO);

    Tensor T = build_and_load(file35, "SO-basis Kinetic Energy Ints", AO);
    Tensor V = build_and_load(file35, "SO-basis Potential Energy Ints", AO);

    H("p,q") = T("p,q") + V("p,q");

    return H;
}

Tensor load_2e(const Dimension& AO)
{
    // psi two-electron integral file
    Tensor g = build("g", AO);
    io::IWL iwl("test.33", tensor::io::kOpenModeOpenExisting);
    io::IWL::read_two(iwl, g);

    return g;
}

int main(int argc, char* argv[])
{
    srand(time(nullptr));
    tensor::initialize(argc, argv);

    // psi checkpoint file
    tensor::io::File file32("test.32", tensor::io::kOpenModeOpenExisting);

    // psi one-electron integral file
    tensor::io::File file35("test.35", tensor::io::kOpenModeOpenExisting);

    // Read information from checkpoint file
    int nirrep = 0;
    file32.read("::Num. irreps", &nirrep, 1);
    printf("nirrep = %d\n", nirrep);
    assert(nirrep == 1);

    int nso = 0;
    file32.read("::Num. SO", &nso, 1);
    printf("nso = %d\n", nso);

    double Enuc = 0.0;
    file32.read("::Nuclear rep. energy", &Enuc, 1);

    // The value to compare to from Psi4.
    double Eref = 0.0;
    file32.read("::SCF energy", &Eref, 1);

    // Define dimension objects
    Dimension AO = {(size_t)nso, (size_t)nso};
    Dimension AO4 = {(size_t)nso, (size_t)nso, (size_t)nso, (size_t)nso};

    // Build tensors
    Tensor S = load_overlap(file35, AO);
    printf("norm of S is %lf\n", S.norm());
    Tensor H = load_1e_hamiltonian(file35, AO);
    Tensor g = load_2e(AO4);

    Tensor Ft = build("Ft", AO);

    Tensor Smhalf = S.power(-0.5);

    Ft("i,j") = Smhalf("mu,i") * Smhalf("nu,j") * H("mu,nu");
    auto Feigen = Ft.syev(kAscending);

    Tensor C = build("C", AO);
    C("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");

    Tensor Cdocc = build("C", {5, (size_t)nso});

    size_t ndocc = 5;
    IndexRange CtoCdocc = { {0,ndocc}, {0,(size_t)nso}};
    //Cdocc.slice(C, CtoCdocc, CtoCdocc);
    Cdocc(CtoCdocc) = C(CtoCdocc);

    // Form initial D
    Tensor D = build("D", AO);
    D("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");

    Tensor F = build("F", AO);

    // start SCF iteration
    bool converged = false;
    double Eelec = 0.0, Eold = 0.0;
    int iter = 1;
    do {
        F("mu,nu") = H("mu,nu");
        F("mu,nu") += D("rho,sigma") * (2.0 * g("mu,nu,rho,sigma") - g("mu,rho,nu,sigma"));
//        F.print(stdout, true);

        // Calculate energy
        Eelec = D("mu,nu") * (H("mu,nu") + F("mu,nu"));
        printf("  @RHF iter %5d: %20.14lf\n", iter++, Enuc + Eelec);

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

        if (std::fabs(Eelec - Eold) < 1.0e-8) converged = true;
        Eold = Eelec;

    } while (!converged);

    tensor::finalize();
    return EXIT_SUCCESS;
}
