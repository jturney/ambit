#include <tensor/tensor.h>
#include <cstdio>

using namespace tensor;

int main(int argc, char* argv[])
{
    tensor::initialize(argc, argv);

    Dimension dims;
    dims.push_back(5); dims.push_back(5);
    Tensor test = Tensor::build(kAgnostic, "Test", dims);
//    test.print(stdout);

    LabeledTensor test_ab1 = test["ab"];
    printf("numdims of test_ab1 = test[\"ab\"] = %zu\n", test_ab1.numdim());
    assert(test_ab1.numdim() == 2);

    LabeledTensor test_ab2 = test("a,b");
    printf("numdims of test_ab2 = test[\"a,b\"] = %zu\n", test_ab2.numdim());
    assert(test_ab2.numdim() == 2);

    LabeledTensor test_ab3 = 2.0 * test_ab1;
    assert(test_ab3.numdim() == 2);

    LabeledTensor test_ab4 = 2.0 * test["ab"];
    assert(test_ab4.numdim() == 2);

    Tensor test2 = Tensor::build(kAgnostic, "Test 2", dims);
    test2["a,b"] = 2.0 * test["ab"];

    tensor::finialize();

    return 0;
}
