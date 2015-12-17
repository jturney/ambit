#include <cstdlib>
#include <ambit/tensor.h>

using namespace ambit;

int main(int argc, char *argv[])
{
    ambit::initialize(argc, argv);

    {
        Tensor A = Tensor::build(CoreTensor, "A", {1000, 1000});
        Tensor B = Tensor::build(CoreTensor, "B", {1000, 1000});
        Tensor C = Tensor::build(CoreTensor, "C", {1000, 1000});

        C("ij") += A("ik") * B("jk");
    }

    ambit::finalize();

    return EXIT_SUCCESS;
}
