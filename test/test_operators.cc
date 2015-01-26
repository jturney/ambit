#include <tensor/tensor.h>
#include <cstdio>

using namespace tensor;

int main(int argc, char* argv[])
{
    tensor::initialize(argc, argv);

    Dimension dims;
    dims.push_back(5); dims.push_back(5);
    Tensor test = Tensor::build(kAgnostic, "Test", dims);
    test.print(stdout, true);

    LabeledTensor test_ab1 = test["ab"];
    assert(test_ab1.numdim() == 2);

    LabeledTensor test_ab2 = test("a,b");
    assert(test_ab2.numdim() == 2);

    LabeledTensor test_ab3 = 2.0 * test_ab1;
    assert(test_ab3.numdim() == 2);
    assert(test_ab3.factor() == 2.0);

    LabeledTensor test_ab4 = 2.0 * test["ab"];
    assert(test_ab4.numdim() == 2);
    assert(test_ab4.factor() == 2.0);

    Tensor test2 = Tensor::build(kAgnostic, "Test 2", dims);
    test2["a,b"] = 2.0 * test["ab"];

    Tensor test3 = Tensor::build(kAgnostic, "Test 3", dims);

    // Element addition test
//    test3["a,b"]  = test["a,b"] + test2["a,b"];
//    test3["a,b"] += test["a,b"] + test2["a,b"];
//    test3["a,b"] -= test["a,b"] + test2["a,b"];
//
//    // Element subtraction test
//    test3["a,b"]  = test["a,b"] - test2["a,b"];
//    test3["a,b"] += test["a,b"] - test2["a,b"];
//    test3["a,b"] -= test["a,b"] - test2["a,b"];

    // Contraction test
    test3["a,b"]  = test["a,c"] * test2["c,b"];
    test3["a,b"] += test["a,c"] * test2["c,b"];
    test3["a,b"] -= test["a,c"] * test2["c,b"];

    tensor::finialize();

    return 0;
}
