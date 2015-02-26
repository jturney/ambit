#include <ambit/tensor.h>

using namespace ambit;

int main(int argc, char* argv[])
{
    Tensor A = Tensor::build(kCore, "A", {1000,1000});
    Tensor B = Tensor::build(kCore, "B", {1000,1000});
    Tensor C = Tensor::build(kCore, "C", {1000,1000});

    C("ij") += A("ik") * B("jk");

    return EXIT_SUCCESS;
}

