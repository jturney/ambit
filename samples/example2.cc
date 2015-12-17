#include <random>

#include <ambit/blocked_tensor.h>

using namespace ambit;

int main(int argc, char *argv[])
{
    ambit::initialize(argc, argv);

    {
        BlockedTensor::add_mo_space("o", "i,j,k,l", {0, 1, 2, 3, 4}, AlphaSpin);
        BlockedTensor::add_mo_space("v", "a,b,c,d", {5, 6, 7, 8, 9}, AlphaSpin);
        BlockedTensor::add_mo_space("O", "I,J,K,L", {0, 1, 2, 3, 4}, BetaSpin);
        BlockedTensor::add_mo_space("V", "A,B,C,D", {5, 6, 7, 8, 9}, BetaSpin);

        BlockedTensor::print_mo_spaces();

        BlockedTensor F = BlockedTensor::build(CoreTensor, "F", {"oo", "ov", "vo"});

        F.iterate([](const std::vector<size_t> & /*indices*/,
                     const std::vector<SpinType> & /*spin*/, double &value)
                  {
                      value = double(std::rand()) / double(RAND_MAX);
                  });

        F.print(stdout);
    }

    ambit::finalize();

    return EXIT_SUCCESS;
}
