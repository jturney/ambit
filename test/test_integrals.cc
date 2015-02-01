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

int main(int argc, char* argv[])
{
    srand(time(NULL));
    tensor::initialize(argc, argv);

    // psi checkpoint file
    tensor::io::File file32("test.32", tensor::io::kOpenModeOpenExisting);

    // psi two-electron integral file
//    tensor::io::File file34("test.34", tensor::io::kOpenModeOpenExisting);

    // psi one-electron integral file
    tensor::io::File file35("test.35", tensor::io::kOpenModeOpenExisting);

    file32.toc().print();
//    file34.toc().print();
    file35.toc().print();

    // Read information from checkpoint file
    int nirrep = 0;
    file32.read("::Num. irreps", &nirrep, 1);
    printf("\n\nnirrep = %d\n", nirrep);

    int nso = 0;
    file32.read("::Num. SO", &nso, 1);
    printf("nso = %d\n", nso);

    int nmo = 0;
    file32.read("::Num. MO's", &nmo, 1);
    printf("nmo = %d\n", nmo);


    // Define dimension objects
    Dimension AO = {(size_t)nso, (size_t)nso};
    Dimension AOvMO = {(size_t)nso, (size_t)nmo};
    Dimension MO = {(size_t)nmo, (size_t)nmo};

    // Build tensors
    Tensor S = load_overlap(file35, AO);
    Tensor H = load_1e_hamiltonian(file35, AO);

    Tensor Ft = build("Ft", AO);

    Tensor Smhalf = S.power(-0.5);
    Smhalf.print(stdout, true);

    Tensor Fh = build("Fh", AO);
    Fh("i,nu") = Smhalf("mu,i") * H("mu,nu");
    Ft("i,j") = Smhalf("nu,j") * Fh("i,nu");

    auto Feigen = Ft.syev(kAscending);
    Feigen["eigenvectors"].print(stdout, true);  // these are transposed into row major.
    Feigen["eigenvalues"].print(stdout, true);

    tensor::finalize();
    return EXIT_SUCCESS;
}
