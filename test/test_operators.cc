#include <tensor/tensor.h>
#include <cstdio>
#include <cmath>
#include <utility>

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

/// Initialize a tensor and a 2-dim matrix with random numbers
void initialize_random_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO]);

std::pair<std::string,double> test_Cij_plus_equal_Aik_Bkj();
std::pair<std::string,double> test_Cij_minus_equal_Aik_Bkj();
std::pair<std::string,double> test_Cij_equal_Aik_Bkj();
std::pair<std::string,double> test_Cij_equal_Aik_Bjk();
std::pair<std::string,double> test_Cij_equal_Aik_Bkj();
std::pair<std::string,double> test_Cij_equal_Aik_Bjk();
std::pair<std::string,double> test_Cijkl_equal_Aijab_Bklab();
std::pair<std::string,double> test_Cikjl_equal_Aijab_Bklab();
std::pair<std::string,double> test_Cij_equal_Aiabc_Bjabc();
std::pair<std::string,double> test_C_equal_A_B(std::string c_ind,std::string a_ind,std::string b_ind,
                                               std::vector<int> c_dim,std::vector<int> a_dim,std::vector<int> b_dim);

double zero = 1.0e-14;

int main(int argc, char* argv[])
{
    srand (time(NULL));
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

    //test3["a,b"]  = test["a,c"] * test2["c,b"] * test2["c,b"];



    std::vector<std::pair<std::string,double> > results;

    results.push_back(test_C_equal_A_B("ij","ik","jk",{0,1},{0,2},{1,2}));
    results.push_back(test_C_equal_A_B("ij","ik","kj",{0,1},{0,2},{2,1}));
    results.push_back(test_C_equal_A_B("ij","ki","jk",{0,1},{2,0},{1,2}));
    results.push_back(test_C_equal_A_B("ij","ki","kj",{0,1},{2,0},{2,1}));
    results.push_back(test_C_equal_A_B("ji","ik","jk",{1,0},{0,2},{1,2}));
    results.push_back(test_C_equal_A_B("ji","ik","kj",{1,0},{0,2},{2,1}));
    results.push_back(test_C_equal_A_B("ji","ki","jk",{1,0},{2,0},{1,2}));
    results.push_back(test_C_equal_A_B("ji","ki","kj",{1,0},{2,0},{2,1}));

    results.push_back(test_Cij_equal_Aik_Bkj());
    results.push_back(test_Cij_equal_Aik_Bjk());

    results.push_back(test_Cij_plus_equal_Aik_Bkj());
    results.push_back(test_Cij_minus_equal_Aik_Bkj());

    results.push_back(test_Cijkl_equal_Aijab_Bklab());
    results.push_back(test_Cij_equal_Aiabc_Bjabc());

    results.push_back(test_Cikjl_equal_Aijab_Bklab());

    tensor::finialize();

    bool success = true;
    for (auto sb : results){
        if (std::fabs(sb.second) > zero) success = false;
    }

    if(true){
        printf("\n\n Summary of tests:");

        printf("\n %-50s %12s %s","Test","Max. error","Result");
        printf("\n %s",std::string(60,'-').c_str());
        for (auto sb : results){
            printf("\n %-50s %7e %s",sb.first.c_str(),sb.second,std::fabs(sb.second) < zero ? "Passed" : "Failed");
        }
        printf("\n %s",std::string(60,'-').c_str());
        printf("\n Tests: %s",success ? "All passed" : "Some failed");
    }

    return success;
}


void initialize_random_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO])
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    double* vec = new double[n0 * n1];
    for (size_t i = 0, ij = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j, ++ij){
            double randnum = double(std::rand())/double(RAND_MAX);
            matrix[i][j] = randnum;
            vec[ij] = randnum;
        }
    }
    tensor.set_data(vec);
    delete[] vec;
}

std::pair<double,double> difference_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO])
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];

    size_t numel = tensor.numel();
    double* result = new double[numel];

    tensor.get_data(result);

    double sum_diff = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0, ij = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j, ++ij){
            double diff = std::fabs(matrix[i][j] - result[ij]);
            sum_diff += diff;
            max_diff = std::max(diff,max_diff);
        }
    }
    delete[] result;
    return std::make_pair(sum_diff,max_diff);
}

void initialize_random_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR])
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    size_t n3 = tensor.dims()[3];

    double* vec = new double[n0 * n1 * n2 * n3];
    for (size_t i = 0, ijkl = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            for (size_t k = 0; k < n2; ++k){
                for (size_t l = 0; l < n3; ++l, ++ijkl){
                    double randnum = double(std::rand())/double(RAND_MAX);
                    matrix[i][j][k][l] = randnum;
                    vec[ijkl] = randnum;
                }
            }
        }
    }
    tensor.set_data(vec);
    delete[] vec;
}

std::pair<double,double> difference_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR])
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    size_t n3 = tensor.dims()[3];

    size_t numel = tensor.numel();

    printf("\n  %zu %zu %zu %zu",n0,n1,n2,n3);
    printf("\n  numel = %zu",numel);

    double* result = new double[numel];

    tensor.get_data(result);

    double sum_diff = 0.0;
    double max_diff = 0.0;

    for (size_t i = 0, ijkl = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            for (size_t k = 0; k < n2; ++k){
                for (size_t l = 0; l < n3; ++l, ++ijkl){
                    double diff = std::fabs(matrix[i][j][k][l] - result[ijkl]);
                    sum_diff += diff;
                    max_diff = std::max(diff,max_diff);
                }
            }
        }
    }
    delete[] result;
    return std::make_pair(sum_diff,max_diff);
}

std::pair<std::string,double> test_C_equal_A_B(std::string c_ind,std::string a_ind,std::string b_ind,
                                               std::vector<int> c_dim,std::vector<int> a_dim,std::vector<int> b_dim)
{
    std::string test = "C(\"" + c_ind + "\") += A(\"" + a_ind + "\") * B(\"" + b_ind + "\")";
    printf("\n  Testing %s",test.c_str());

    std::vector<int> dims;
    dims.push_back(9);
    dims.push_back(6);
    dims.push_back(7);

    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;

    Dimension dimsA;
    dimsA.push_back(dims[a_dim[0]]); dimsA.push_back(dims[a_dim[1]]);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsB;
    dimsB.push_back(dims[b_dim[0]]); dimsB.push_back(dims[b_dim[1]]);
    Tensor B = Tensor::build(kCore, "B", dimsB);

    Dimension dimsC;
    dimsC.push_back(dims[c_dim[0]]); dimsC.push_back(dims[c_dim[1]]);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    initialize_random_2(B,b2);
    std::pair<double,double> b_diff = difference_2(B,b2);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C(c_ind) += A(a_ind) * B(b_ind);

    std::vector<int> n(3);
    for (n[0] = 0; n[0] < ni; ++n[0]){
        for (n[1] = 0; n[1] < nj; ++n[1]){
            for (n[2] = 0; n[2] < nk; ++n[2]){
                int aind1 = n[a_dim[0]];
                int aind2 = n[a_dim[1]];
                int bind1 = n[b_dim[0]];
                int bind2 = n[b_dim[1]];
                int cind1 = n[c_dim[0]];
                int cind2 = n[c_dim[1]];
                c2[cind1][cind2] += a2[aind1][aind2] * b2[bind1][bind2];
            }
        }
    }
    c_diff = difference_2(C,c2);

    return std::make_pair(test,c_diff.second);
}

std::pair<std::string,double> test_Cij_plus_equal_Aik_Bkj()
{
    std::string test = "C(\"ij\") += A(\"ik\") * B(\"kj\")";
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

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    initialize_random_2(B,b2);
    std::pair<double,double> b_diff = difference_2(B,b2);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C("ij") += A("ik") * B("kj");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                c2[i][j] += a2[i][k] * b2[k][j];
            }
        }
    }
    c_diff = difference_2(C,c2);

    return std::make_pair(test,c_diff.second);
}


std::pair<std::string,double> test_Cij_minus_equal_Aik_Bkj()
{
    std::string test = "C(\"ij\") -= A(\"ik\") * B(\"kj\")";
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

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    initialize_random_2(B,b2);
    std::pair<double,double> b_diff = difference_2(B,b2);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C("ij") -= A("ik") * B("kj");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                c2[i][j] -= a2[i][k] * b2[k][j];
            }
        }
    }
    c_diff = difference_2(C,c2);

    return std::make_pair(test,c_diff.second);
}



std::pair<std::string,double> test_Cij_equal_Aik_Bkj()
{
    std::string test = "C(\"ij\") = A(\"ik\") * B(\"kj\")";
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

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    initialize_random_2(B,b2);
    std::pair<double,double> b_diff = difference_2(B,b2);

    C.zero();
    C("ij") = A("ik") * B("kj");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k){
                c2[i][j] += a2[i][k] * b2[k][j];
            }
        }
    }
    std::pair<double,double> c_diff = difference_2(C,c2);

    return std::make_pair(test,c_diff.second);
}

std::pair<std::string,double> test_Cij_equal_Aik_Bjk()
{
    std::string test = "C(\"ij\") = A(\"ik\") * B(\"jk\")";
    printf("\n  Testing %s",test.c_str());

    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(nk);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsB;
    dimsB.push_back(nj); dimsB.push_back(nk);
    Tensor B = Tensor::build(kCore, "B", dimsB);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    initialize_random_2(B,b2);
    std::pair<double,double> b_diff = difference_2(B,b2);

    C.zero();
    C("ij") = A("ik") * B("jk");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k){
                c2[i][j] += a2[i][k] * b2[j][k];
            }
        }
    }
    std::pair<double,double> c_diff = difference_2(C,c2);

    return std::make_pair(test,c_diff.second);
}


std::pair<std::string,double> test_Cijkl_equal_Aijab_Bklab()
{
    std::string test = "C(\"ijkl\") += A(\"ijab\") * B(\"klab\")";
    printf("\n  Testing %s",test.c_str());

    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(nj);
    dimsA.push_back(na); dimsA.push_back(nb);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsB;
    dimsB.push_back(nk); dimsB.push_back(nl);
    dimsB.push_back(na); dimsB.push_back(nb);
    Tensor B = Tensor::build(kCore, "B", dimsB);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    dimsC.push_back(nk); dimsC.push_back(nl);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_4(A,a4);
    std::pair<double,double> a_diff = difference_4(A,a4);

    initialize_random_4(B,b4);
    std::pair<double,double> b_diff = difference_4(B,b4);

    initialize_random_4(C,c4);
    std::pair<double,double> c_diff = difference_4(C,c4);

    C("ijkl") += A("ijab") * B("klab");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    for (size_t a = 0; a < na; ++a){
                        for (size_t b = 0; b < nb; ++b){
                            c4[i][j][k][l] += a4[i][j][a][b] * b4[k][l][a][b];
                        }
                    }
                }
            }
        }
    }
    c_diff = difference_4(C,c4);

    return std::make_pair(test,c_diff.second);
}


std::pair<std::string,double> test_Cikjl_equal_Aijab_Bklab()
{
    std::string test = "C(\"ikjl\") += A(\"ijab\") * B(\"klab\")";
    printf("\n  Testing %s",test.c_str());

    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(nj);
    dimsA.push_back(na); dimsA.push_back(nb);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsB;
    dimsB.push_back(nk); dimsB.push_back(nl);
    dimsB.push_back(na); dimsB.push_back(nb);
    Tensor B = Tensor::build(kCore, "B", dimsB);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nk);
    dimsC.push_back(nj); dimsC.push_back(nl);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_4(A,a4);
    std::pair<double,double> a_diff = difference_4(A,a4);

    initialize_random_4(B,b4);
    std::pair<double,double> b_diff = difference_4(B,b4);

    initialize_random_4(C,c4);
    std::pair<double,double> c_diff = difference_4(C,c4);

    C("ijkl") += A("ijab") * B("klab");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    for (size_t a = 0; a < na; ++a){
                        for (size_t b = 0; b < nb; ++b){
                            c4[i][k][j][l] += a4[i][j][a][b] * b4[k][l][a][b];
                        }
                    }
                }
            }
        }
    }
    c_diff = difference_4(C,c4);

    return std::make_pair(test,c_diff.second);
}



std::pair<std::string,double> test_Cij_equal_Aiabc_Bjabc()
{
    std::string test = "C(\"ij\") += A(\"iabc\") * B(\"jabc\")";
    printf("\n  Testing %s",test.c_str());

    size_t ni = 9;
    size_t nj = 6;
    size_t na = 6;
    size_t nb = 7;
    size_t nc = 8;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(na);
    dimsA.push_back(nb); dimsA.push_back(nc);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsB;
    dimsB.push_back(nj); dimsB.push_back(na);
    dimsB.push_back(nb); dimsB.push_back(nc);
    Tensor B = Tensor::build(kCore, "B", dimsB);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_4(A,a4);
    std::pair<double,double> a_diff = difference_4(A,a4);

    initialize_random_4(B,b4);
    std::pair<double,double> b_diff = difference_4(B,b4);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C("ij") += A("iabc") * B("jabc");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t a = 0; a < na; ++a){
                for (size_t b = 0; b < nb; ++b){
                    for (size_t c = 0; c < nc; ++c){
                        c2[i][j] += a4[i][a][b][c] * b4[j][a][b][c];
                    }
                }
            }
        }
    }
    c_diff = difference_2(C,c2);

    return std::make_pair(test,c_diff.second);
}
