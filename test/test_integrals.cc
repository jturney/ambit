#include <tensor/tensor.h>
#include <tensor/io/io.h>

#include <string>

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
    srand(time(NULL));
    tensor::initialize(argc, argv);

    // psi checkpoint file
    tensor::io::File file32("test.32", tensor::io::kOpenModeOpenExisting);


    // psi one-electron integral file
    tensor::io::File file35("test.35", tensor::io::kOpenModeOpenExisting);

//    file32.toc().print();
//    file34.toc().print();
//    file35.toc().print();

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
    Tensor H = load_1e_hamiltonian(file35, AO);
    Tensor g = load_2e(AO4);

//    g.print(stdout, false);

    Tensor Ft = build("Ft", AO);

    Tensor Smhalf = S.power(-0.5);
//    Smhalf.print(stdout, true);

    Ft("i,j") = Smhalf("mu,i") * Smhalf("nu,j") * H("mu,nu");
    Ft.print(stdout, true);

    auto Feigen = Ft.syev(kAscending);

    Tensor C = build("C", AO);
    C("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");

    Tensor Cdocc = build("C", {5, (size_t)nso});
    double *data = new double[Cdocc.numel()];
    IndexRange Cdocc_range = { std::make_pair(0, 5), std::make_pair(0, nso) };
    C.get_data(data, Cdocc_range);
    Cdocc.set_data(data);

    C.print(stdout, true);
    Cdocc.print(stdout, true);

    // Form initial D
    Tensor D = build("D", AO);
    D("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");
    D.print(stdout, true);

    Tensor F = build("F", AO);

    // start SCF iteration
    F("mu,nu") = H("mu,nu");
    F("mu,nu") += D("rho,sigma") * (2.0 * g("mu,nu,rho,sigma") - g("mu,rho,nu,sigma"));
    F.print(stdout, true);

    F("mu,nu") = H("mu,nu");
    F("mu,nu") += 2.0 * D("rho,sigma") * g("mu,nu,rho,sigma");
//    F("mu,nu") -= D("rho,sigma") * g("mu,rho,nu,sigma");
    F.print(stdout, true);

    tensor::finalize();
    return EXIT_SUCCESS;
}
