#include <tensor/tensor.h>
//#include <cstdio>
#include <cmath>
//#include <utility>

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

double test_Cij_plus_equal_Aik_Bkj();
double test_Cij_minus_equal_Aik_Bkj();
double test_Cij_equal_Aik_Bkj();
double test_Cij_equal_Aik_Bjk();
double test_Cij_equal_Aik_Bkj();
double test_Cij_equal_Aik_Bjk();
double test_Cijkl_equal_Aijab_Bklab();
double test_Cikjl_equal_Aijab_Bklab();
double test_Cij_equal_Aiabc_Bjabc();
double test_Cij_equal_Aji();
double test_Cijkl_equal_Akilj();
double test_Cijkl_equal_Akijl();
double test_C_equal_2_A();
double test_C_plus_equal_2_A();
double test_C_minus_equal_2_A();
double test_C_times_equal_2();
double test_C_divide_equal_2();
double test_Cij_equal_Cij();
double test_Cilkj_equal_Aibaj_Bblak();
double test_Cljik_equal_Abija_Blbak();

std::pair<std::string,double> test_C_equal_A_B(std::string c_ind,std::string a_ind,std::string b_ind,
                                               std::vector<int> c_dim,std::vector<int> a_dim,std::vector<int> b_dim);

double zero = 1.0e-12;

int main(int argc, char* argv[])
{
    srand (time(NULL));
    tensor::initialize(argc, argv);

    auto test_functions = {
            std::make_pair(test_C_equal_2_A,               "C(\"ij\") = 2.0 * A(\"ij\")"),
            std::make_pair(test_C_plus_equal_2_A,          "C(\"ij\") += 2.0 * A(\"ij\")"),
            std::make_pair(test_C_minus_equal_2_A,         "C(\"ij\") -= 2.0 * A(\"ij\")"),
            std::make_pair(test_C_times_equal_2,           "C(\"ij\") *= 2.0"),
            std::make_pair(test_C_divide_equal_2,          "C(\"ij\") /= 2.0"),
            std::make_pair(test_Cij_equal_Aik_Bkj,         "C(\"ij\") = A(\"ik\") * B(\"kj\")"),
            std::make_pair(test_Cij_equal_Aik_Bjk,         "C(\"ij\") = A(\"ik\") * B(\"jk\")"),
            std::make_pair(test_Cij_plus_equal_Aik_Bkj,    "C(\"ij\") += A(\"ik\") * B(\"kj\")"),
            std::make_pair(test_Cij_minus_equal_Aik_Bkj,   "C(\"ij\") -= A(\"ik\") * B(\"kj\")"),
            std::make_pair(test_Cijkl_equal_Aijab_Bklab,   "C(\"ijkl\") += A(\"ijab\") * B(\"klab\")"),
            std::make_pair(test_Cij_equal_Aiabc_Bjabc,     "C(\"ij\") += A(\"iabc\") * B(\"jabc\")"),
            std::make_pair(test_Cikjl_equal_Aijab_Bklab,   "C(\"ikjl\") += A(\"ijab\") * B(\"klab\")"),            
            std::make_pair(test_Cij_equal_Aji,             "C(\"ij\") = A(\"ji\")"),
            std::make_pair(test_Cijkl_equal_Akilj,         "C(\"ijkl\") = A(\"kilj\")"),
            std::make_pair(test_Cijkl_equal_Akijl,         "C(\"ijkl\") = A(\"kijl\")"),
            std::make_pair(test_Cij_equal_Cij,             "C(\"ij\") = C(\"ji\") not allowed"),
            std::make_pair(test_Cilkj_equal_Aibaj_Bblak,   "C(\"ilkj\") += A(\"ibaj\") * B(\"blak\")"),
            std::make_pair(test_Cljik_equal_Abija_Blbak,   "C(\"ljik\") += A(\"bija\") * B(\"lbak\")")
    };

    std::vector<std::pair<std::string,double> > results;

    for (auto test_function : test_functions) {
        printf("  Testing %s\n", test_function.second);
        try {
            results.push_back(std::make_pair(test_function.second, test_function.first()));
        }
        catch (std::exception& e) {
            printf("    Exception caught: %s\n", e.what());
        }
    }

    results.push_back(test_C_equal_A_B("ij","ik","jk",{0,1},{0,2},{1,2}));
    results.push_back(test_C_equal_A_B("ij","ik","kj",{0,1},{0,2},{2,1}));
    results.push_back(test_C_equal_A_B("ij","ki","jk",{0,1},{2,0},{1,2}));
    results.push_back(test_C_equal_A_B("ij","ki","kj",{0,1},{2,0},{2,1}));
    results.push_back(test_C_equal_A_B("ji","ik","jk",{1,0},{0,2},{1,2}));
    results.push_back(test_C_equal_A_B("ji","ik","kj",{1,0},{0,2},{2,1}));
    results.push_back(test_C_equal_A_B("ji","ki","jk",{1,0},{2,0},{1,2}));
    results.push_back(test_C_equal_A_B("ji","ki","kj",{1,0},{2,0},{2,1}));

    tensor::finalize();

    bool success = true;
    for (auto sb : results){
        if (std::fabs(sb.second) > zero) success = false;
    }

    if(true){
        printf("\n\n Summary of tests:");

        printf("\n %-50s %12s %s","Test","Max. error","Result");
        printf("\n %s",std::string(70,'-').c_str());
        for (auto sb : results){
            printf("\n %-50s %7e %s",sb.first.c_str(),sb.second,std::fabs(sb.second) < zero ? "Passed" : "Failed");
        }
        printf("\n %s",std::string(70,'-').c_str());
        printf("\n Tests: %s\n",success ? "All passed" : "Some failed");
    }

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
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
    printf("  Testing %s\n",test.c_str());

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

double test_Cij_plus_equal_Aik_Bkj()
{
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

    return c_diff.second;
}


double test_Cij_minus_equal_Aik_Bkj()
{
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

    return c_diff.second;
}

double test_Cij_equal_Aik_Bkj()
{
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

    return c_diff.second;
}

double test_Cij_equal_Aik_Bjk()
{
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

    return c_diff.second;
}


double test_Cijkl_equal_Aijab_Bklab()
{
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

    return c_diff.second;
}


double test_Cikjl_equal_Aijab_Bklab()
{
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

    C("ikjl") += A("ijab") * B("klab");

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

    return c_diff.second;
}

double test_Cij_equal_Aiabc_Bjabc()
{
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

    return c_diff.second;
}

double test_C_equal_2_A()
{
    size_t ni = 9;
    size_t nj = 6;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(nj);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C("ij") = 2.0 * A("ij");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = 2.0 * a2[i][j];
        }
    }
    c_diff = difference_2(C,c2);

    return c_diff.second;
}

double test_C_plus_equal_2_A()
{
    size_t ni = 9;
    size_t nj = 6;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(nj);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C("ij") += 2.0 * A("ij");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] += 2.0 * a2[i][j];
        }
    }
    c_diff = difference_2(C,c2);

    return c_diff.second;
}

double test_C_minus_equal_2_A()
{
    size_t ni = 9;
    size_t nj = 6;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(nj);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C("ij") -= 2.0 * A("ij");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] -= 2.0 * a2[i][j];
        }
    }
    c_diff = difference_2(C,c2);

    return c_diff.second;
}

double test_C_times_equal_2()
{
    size_t ni = 9;
    size_t nj = 6;

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C("ij") *= 2.0;

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] *= 2.0;
        }
    }
    c_diff = difference_2(C,c2);

    return c_diff.second;
}

double test_C_divide_equal_2()
{
    size_t ni = 9;
    size_t nj = 6;

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    C("ij") /= 2.0;

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] /= 2.0;
        }
    }
    c_diff = difference_2(C,c2);

    return c_diff.second;
}

double test_Cij_equal_Aji()
{
    size_t ni = 9;
    size_t nj = 6;

    Dimension dimsA;
    dimsA.push_back(nj); dimsA.push_back(ni);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_2(C,c2);
    std::pair<double,double> c_diff = difference_2(C,c2);

    initialize_random_2(A,a2);
    std::pair<double,double> a_diff = difference_2(A,a2);

    C("ij") = A("ji");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = a2[j][i];
        }
    }
    c_diff = difference_2(C,c2);

    //A.print(stdout, true);
    //C.print(stdout, true);

    return c_diff.second;
}
double test_Cijkl_equal_Akijl()
{
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 5;
    size_t nl = 4;

    Dimension dimsA = {nk,ni,nj,nl};
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsC = {ni,nj,nk,nl};
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_4(C,c4);
    std::pair<double,double> c_diff = difference_4(C,c4);

    initialize_random_4(A,a4);
    std::pair<double,double> a_diff = difference_4(A,a4);

    C("ijkl") = A("kijl");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[i][j][k][l] = a4[k][i][j][l];
                }
            }
        }
    }
    c_diff = difference_4(C,c4);

    //A.print(stdout, true);
    //C.print(stdout, true);

    return c_diff.second;
}
double test_Cijkl_equal_Akilj()
{
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 5;
    size_t nl = 4;

    Dimension dimsA = {nk,ni,nl,nj};
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsC = {ni,nj,nk,nl};
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_4(C,c4);
    std::pair<double,double> c_diff = difference_4(C,c4);

    initialize_random_4(A,a4);
    std::pair<double,double> a_diff = difference_4(A,a4);

    C("ijkl") = A("kilj");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[i][j][k][l] = a4[k][i][l][j];
                }
            }
        }
    }
    c_diff = difference_4(C,c4);

    //A.print(stdout, true);
    //C.print(stdout, true);

    return c_diff.second;
}

double test_Cij_equal_Cij()
{
    size_t ni = 9;
    size_t nj = 6;

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    try {
        C("ij") = C("ij");
    }
    catch (std::exception& e) {
        return 0.00;
    }
    return 1.0;
}

double test_Cilkj_equal_Aibaj_Bblak()
{
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;

    Dimension dimsA;
    dimsA.push_back(ni); dimsA.push_back(nb);
    dimsA.push_back(na); dimsA.push_back(nj);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsB;
    dimsB.push_back(nb); dimsB.push_back(nl);
    dimsB.push_back(na); dimsB.push_back(nk);
    Tensor B = Tensor::build(kCore, "B", dimsB);

    Dimension dimsC;
    dimsC.push_back(ni); dimsC.push_back(nl);
    dimsC.push_back(nk); dimsC.push_back(nj);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_4(A,a4);
    std::pair<double,double> a_diff = difference_4(A,a4);

    initialize_random_4(B,b4);
    std::pair<double,double> b_diff = difference_4(B,b4);

    initialize_random_4(C,c4);
    std::pair<double,double> c_diff = difference_4(C,c4);

    C("ilkj") += A("ibaj") * B("blak");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    for (size_t a = 0; a < na; ++a){
                        for (size_t b = 0; b < nb; ++b){
                            c4[i][l][k][j] += a4[i][b][a][j] * b4[b][l][a][k];
                        }
                    }
                }
            }
        }
    }
    c_diff = difference_4(C,c4);

    return c_diff.second;
}

double test_Cljik_equal_Abija_Blbak()
{
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;

    Dimension dimsA;
    dimsA.push_back(nb); dimsA.push_back(ni);
    dimsA.push_back(nj); dimsA.push_back(na);
    Tensor A = Tensor::build(kCore, "A", dimsA);

    Dimension dimsB;
    dimsB.push_back(nl); dimsB.push_back(nb);
    dimsB.push_back(na); dimsB.push_back(nk);
    Tensor B = Tensor::build(kCore, "B", dimsB);

    Dimension dimsC;
    dimsC.push_back(nl); dimsC.push_back(nj);
    dimsC.push_back(ni); dimsC.push_back(nk);
    Tensor C = Tensor::build(kCore, "C", dimsC);

    initialize_random_4(A,a4);
    std::pair<double,double> a_diff = difference_4(A,a4);

    initialize_random_4(B,b4);
    std::pair<double,double> b_diff = difference_4(B,b4);

    initialize_random_4(C,c4);
    std::pair<double,double> c_diff = difference_4(C,c4);

    C("ljik") += A("bija") * B("lbak");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    for (size_t a = 0; a < na; ++a){
                        for (size_t b = 0; b < nb; ++b){
                            c4[l][j][i][k] += a4[b][i][j][a] * b4[l][b][a][k];
                        }
                    }
                }
            }
        }
    }
    c_diff = difference_4(C,c4);

    return c_diff.second;
}

