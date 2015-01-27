#include <tensor/tensor.h>
#include <cstdio>

#define MAXTWO 10
#define MAXFOUR 10

double a2[MAXTWO][MAXTWO];
double b2[MAXTWO][MAXTWO];
double c2[MAXTWO][MAXTWO];
double a4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double b4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double c4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double d4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];

using namespace tensor;

/// Initialize a tensor and a double* with the same sequence of
/// random numbers
void initialize_random(Tensor& tensor,double* vec);

std::pair<std::string,double> test_Cij_Aik_Bjk();



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

    test3["a,b"]  = test["a,c"] * test2["c,b"] * test2["c,b"];



    std::vector<std::pair<std::string,double> > results;

    results.push_back(test_Cij_Aik_Bjk());

    tensor::finialize();

    return 0;
}

void initialize_random(Tensor& tensor,double* vec)
{
    size_t numel = tensor.numel();
    for (size_t i = 0; i < numel; ++i){
        double randnum = 1.0 - 2.0 * double(std::rand())/double(RAND_MAX);
        vec[i] = randnum;
    }
    tensor.set_data(vec);
}


std::pair<std::string,double> test_Cij_Aik_Bkj()
{
    std::string test = "C2(\"ij\") += A2(\"ik\") * B2(\"kj\")";
    printf("\n  Testing %s",test.c_str());

    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(nk);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsB;
    dimsB.push_back(nk); dimsB.push_back(nj);
    Tensor B = Tensor::build(kCore, "B", dimsB);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random(A,a2);
    std::pair<double,double> a_diff = difference(A,a2);
    printf("\n A error: sum = %e max = %e",a_diff.first,a_diff.second);

    initialize_random(B,b2);
    std::pair<double,double> b_diff = difference(B,b2);
    printf("\n B2 error: sum = %e max = %e",b_diff.first,b_diff.second);

    C.zero();
    C("ij") += A("ik") * B("kj");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k){
                c2[i][j] += a2[i][k] * b2[k][j];
            }
        }
    }
    std::pair<double,double> C_diff = difference_2(C2,c2);
    outfile->Printf("\n C(p,q) error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,C_diff.second);
}
