#include <tensor/tensor.h>

using namespace tensor;

int main(int argc, char* argv[])
{
    Tensor A = Tensor::build(kCore, "A", {10000,10000});
    Tensor B = Tensor::build(kCore, "B", {10000,10000});
    Tensor C = Tensor::build(kCore, "C", {10000,10000});

    C("ij") += A("ik") * B("jk");

    return EXIT_SUCCESS;
}

