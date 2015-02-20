#include <blocked_tensor/blocked_tensor.h>
#include <cmath>
#include <cstdlib>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define MAXTWO 10
#define MAXFOUR 10

double a1[MAXTWO];
double a2[MAXTWO][MAXTWO];
double b2[MAXTWO][MAXTWO];
double c2[MAXTWO][MAXTWO];
double d2[MAXTWO][MAXTWO];
double a4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double b4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double c4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double d4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];

using namespace tensor;

/// Expected relative accuracy for numerical exactness
const double epsilon = 1.0E-14;
const double zero = 1.0E-14;

/// Scheme to categorize expected vs. actual op behavior
enum TestResult {
    kPass,
    kFail,
    kException
};

/// Global alpha parameter
double alpha = 1.0;
/// Global beta parameter
double beta = 0.0;
/// 0 - explicit call, 1 - = OO, 2 - += OO, 3 - -= OO
int mode = 0;

TensorType tensor_type = kCore;

void initialize_random(Tensor &tensor, double matrix[MAXTWO])
{
    size_t n0 = tensor.dims()[0];
    aligned_vector<double>& vec = tensor.data();
    for (size_t i = 0; i < n0; ++i){
        double randnum = double(std::rand())/double(RAND_MAX);
        matrix[i] = randnum;
        vec[i] = randnum;
    }
}

void initialize_random(Tensor &tensor, double matrix[MAXTWO][MAXTWO])
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    aligned_vector<double>& vec = tensor.data();
    for (size_t i = 0, ij = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j, ++ij){
            double randnum = double(std::rand())/double(RAND_MAX);
            matrix[i][j] = randnum;
            vec[ij] = randnum;
        }
    }
}

void initialize_random(Tensor &tensor, double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR])
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    size_t n3 = tensor.dims()[3];

    aligned_vector<double>& vec = tensor.data();
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
}

std::pair<double,double> difference(Tensor &tensor, double matrix[MAXTWO])
{
    size_t n0 = tensor.dims()[0];

    const aligned_vector<double>& result = tensor.data();

    double sum_diff = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < n0; ++i){
        double diff = std::fabs(matrix[i] - result[i]);
        sum_diff += diff;
        max_diff = std::max(diff,max_diff);
    }
    return std::make_pair(sum_diff,max_diff);
}

std::pair<double,double> difference(Tensor &tensor, double matrix[MAXTWO][MAXTWO])
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];

    const aligned_vector<double>& result = tensor.data();

    double sum_diff = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0, ij = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j, ++ij){
            double diff = std::fabs(matrix[i][j] - result[ij]);
            sum_diff += diff;
            max_diff = std::max(diff,max_diff);
        }
    }
    return std::make_pair(sum_diff,max_diff);
}

std::pair<double,double> difference(Tensor &tensor, double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR])
{
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    size_t n3 = tensor.dims()[3];

    const aligned_vector<double>& result = tensor.data();

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
    return std::make_pair(sum_diff,max_diff);
}

Tensor build_and_fill(const std::string& name, const Dimension& dims, double matrix[MAXTWO])
{
    Tensor T = Tensor::build(tensor_type, name, dims);
    initialize_random(T, matrix);
    std::pair<double,double> a_diff = difference(T, matrix);
    if (std::fabs(a_diff.second) > zero) throw std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

Tensor build_and_fill(const std::string& name, const Dimension& dims, double matrix[MAXTWO][MAXTWO])
{
    Tensor T = Tensor::build(tensor_type, name, dims);
    initialize_random(T, matrix);
    std::pair<double,double> a_diff = difference(T, matrix);
    if (std::fabs(a_diff.second) > zero) throw std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

Tensor build_and_fill(const std::string& name, const Dimension& dims, double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR])
{
    Tensor T = Tensor::build(tensor_type, name, dims);
    initialize_random(T, matrix);
    std::pair<double,double> a_diff = difference(T, matrix);
    if (std::fabs(a_diff.second) > zero) throw std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

double test_mo_space()
{
    MOSpace alpha_occ("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    MOSpace alpha_vir("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    return 0.0;
}

double test_add_mo_space()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s,t",{"o","v"});
    return 0.0;
}

double test_add_mo_space_nonexisting_space()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s,t",{"o","v"});
    return 0.0;
}

double test_add_mo_space_repeated_index1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,i",{5,6,7,8,9},AlphaSpin);
    return 0.0;
}

double test_add_mo_space_repeated_index2()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,a",{5,6,7,8,9},AlphaSpin);
    return 0.0;
}

double test_add_mo_space_repeated_index3()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s,c",{"o","v"});
    return 0.0;
}

double test_add_mo_space_no_name1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("","i,j,k",{0,1,2,3,4},AlphaSpin);
    return 0.0;
}

double test_add_mo_space_no_name2()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("","p,q,r,s",{"o","v"});
    return 0.0;
}

double test_add_mo_space_no_index1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","",{0,1,2,3,4},AlphaSpin);
    return 0.0;
}

double test_add_mo_space_no_index2()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","",{"o","v"});
    return 0.0;
}

double test_add_mo_space_no_mos()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{},AlphaSpin);
    return 0.0;
}

double test_add_mo_space_repeated_space1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("o","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    return 0.0;
}

double test_add_mo_space_repeated_space2()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("o","p,q,r,s,c",{"o","v"});
    return 0.0;
}

double test_block_creation1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::build(kCore,"T",{"oo","vv"});
    return 0.0;
}

double test_block_creation2()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});
    BlockedTensor::build(kCore,"F",{"gg"});
    BlockedTensor::build(kCore,"V",{"gggg"});
    return 0.0;
}

double test_block_creation3()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("c","m,n",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("a","u,v",{3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","e,f",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("h","i,j,k,l",{"c","a"});
    BlockedTensor::add_composite_mo_space("p","a,b,c,d",{"a","v"});
    BlockedTensor::build(kCore,"T1",{"hp"});
    BlockedTensor::build(kCore,"T2",{"hhpp"});
    return 0.0;
}

double test_block_creation_bad_rank()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::build(kCore,"T",{"oo","ovv"});
    return 0.0;
}

double test_block_norm_1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("a","u,v",{2,3,4},NoSpin);
    BlockedTensor::add_mo_space("v","e,f",{5,6,7,8,9},NoSpin);
    BlockedTensor T2 = BlockedTensor::build(kCore,"T2",{"aavv"});
    T2.set(0.5);
    double diff = T2.norm(1) - 112.5;
    return diff;
}

double test_block_norm_2()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("a","u,v",{2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","e,f",{5,6,7,8,9},AlphaSpin);
    BlockedTensor T2 = BlockedTensor::build(kCore,"T2",{"aavv"});
    T2.set(0.5);
    double diff = T2.norm(2) - 7.5;
    return diff;
}

double test_block_norm_3()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("a","u,v",{2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","e,f",{5,6,7,8,9},AlphaSpin);
    BlockedTensor T2 = BlockedTensor::build(kCore,"T2",{"aavv"});
    T2.set(0.5);
    double diff = T2.norm(0) - 0.5;
    return diff;
}

double test_block_zero()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("a","u,v",{2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","e,f",{5,6,7,8,9},AlphaSpin);
    BlockedTensor T2 = BlockedTensor::build(kCore,"T2",{"aavv"});
    T2.set(0.5);
    T2.zero();
    return T2.norm(2);
}

double test_block_scale()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("a","u,v",{2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","e,f",{5,6,7,8,9},AlphaSpin);
    BlockedTensor T2 = BlockedTensor::build(kCore,"T2",{"aavv"});
    T2.set(2.0);
    T2.scale(0.25);
    double diff = T2.norm(2) - 7.5;
    return diff;
}

double test_block_labels1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor T = BlockedTensor::build(kCore,"T",{"oo","vv"});
    T("ij");
    return 0.0;
}

double test_block_retrive_block1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor T = BlockedTensor::build(kCore,"T",{"oo","vv"});
    T.block("oo");
    return 0.0;
}

double test_block_retrive_block2()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});
    BlockedTensor T = BlockedTensor::build(kCore,"T",{"oo","vv"});
    T.block("og");
    return 0.0;
}

double test_block_retrive_block3()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor T = BlockedTensor::build(kCore,"T",{"oo","vv"});
    T.block("ov");
    return 0.0;
}

double test_block_retrive_block4()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});
    BlockedTensor T = BlockedTensor::build(kCore,"T",{"oo","vv"});
    T.block("");
    return 0.0;
}

double test_Cij_equal_Aji()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"oo","vv","ov","vo"});
    BlockedTensor C = BlockedTensor::build(kCore,"C",{"oo","vv","ov","vo"});

    Tensor Aoo = A.block("oo");
    Tensor Coo = C.block("oo");

    size_t no = 3;
    size_t nv = 5;

    Tensor Aoo_t = build_and_fill("A", {no, no}, a2);
    Tensor Coo_t = build_and_fill("C", {no, no}, c2);

    Aoo("ij") = Aoo_t("ij");
    Coo("ij") = Coo_t("ij");

    C("ij") = A("ji");

    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            c2[i][j] = a2[j][i];
        }
    }

    double diff_oo = difference(Coo, c2).second;

    Tensor Aov = A.block("ov");
    Tensor Cvo = C.block("vo");

    Tensor Aov_t = build_and_fill("A", {no, nv}, a2);
    Tensor Cvo_t = build_and_fill("C", {nv, no}, c2);

    Aov("ij") = Aov_t("ij");
    Cvo("ij") = Cvo_t("ij");

    C("ai") = A("ia");

    for (size_t i = 0; i < no; ++i){
        for (size_t a = 0; a < nv; ++a){
            c2[a][i] = a2[i][a];
        }
    }

    double diff_vo = difference(Cvo, c2).second;

    Tensor Avv = A.block("vv");
    Tensor Cvv = C.block("vv");

    Tensor Avv_t = build_and_fill("A", {nv, nv}, a2);
    Tensor Cvv_t = build_and_fill("C", {nv, nv}, c2);

    Avv("ij") = Avv_t("ij");
    Cvv("ij") = Cvv_t("ij");

    C("ab") = A("ab");

    for (size_t a = 0; a < nv; ++a){
        for (size_t b = 0; b < nv; ++b){
            c2[a][b] = a2[a][b];
        }
    }

    double diff_vv = difference(Cvv, c2).second;

    return diff_oo + diff_vo + diff_vv;
}

double test_Cijab_plus_equal_Aaibj()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"vovo","ovvo","voov"});
    BlockedTensor C = BlockedTensor::build(kCore,"C",{"oovv","ovvo","voov"});

    size_t no = 3;
    size_t nv = 5;

    Tensor Avovo = A.block("vovo");
    Tensor Coovv = C.block("oovv");

    Tensor Avovo_t = build_and_fill("A", {nv, no, nv, no}, a4);
    Tensor Coovv_t = build_and_fill("C", {no, no, nv, nv}, c4);

    Avovo("pqrs") = Avovo_t("pqrs");
    Coovv("pqrs") = Coovv_t("pqrs");

    C("ijab") += A("aibj");

    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            for (size_t a = 0; a < nv; ++a){
                for (size_t b = 0; b < nv; ++b){
                    c4[i][j][a][b] += a4[a][i][b][j];
                }
            }
        }
    }

    return difference(Coovv, c4).second;
}

double test_Cbija_minus_equal_Ajabi()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"vovo","ovvo","voov"});
    BlockedTensor C = BlockedTensor::build(kCore,"C",{"oovv","ovvo","voov"});

    size_t no = 3;
    size_t nv = 5;

    Tensor Aovvo = A.block("ovvo");
    Tensor Cvoov = C.block("voov");

    Tensor Aovvo_t = build_and_fill("A", {no, nv, nv, no}, a4);
    Tensor Cvoov_t = build_and_fill("C", {nv, no, no, nv}, c4);

    Aovvo("pqrs") = Aovvo_t("pqrs");
    Cvoov("pqrs") = Cvoov_t("pqrs");

    C("bija") -= A("jabi");

    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            for (size_t a = 0; a < nv; ++a){
                for (size_t b = 0; b < nv; ++b){
                    c4[b][i][j][a] -= a4[j][a][b][i];
                }
            }
        }
    }

    return difference(Cvoov, c4).second;
}

double test_Cij_times_equal_double()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"oo","vv","ov","vo"});

    size_t no = 3;
    size_t nv = 5;

    Tensor Aoo_t = build_and_fill("Aoo", {no, no}, a2);
    Tensor Aov_t = build_and_fill("Aov", {no, nv}, b2);

    A.block("oo")("pq") = Aoo_t("pq");
    A.block("ov")("pq") = Aov_t("pq");

    A("ij") *= std::exp(1.0);

    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            c2[i][j] = std::exp(1.0) * a2[i][j];
        }
    }

    Tensor Aoo = A.block("oo");
    double diff_oo = difference(Aoo, c2).second;

//    A("ia") /= std::exp(1.0);

    for (size_t i = 0; i < no; ++i){
        for (size_t a = 0; a < nv; ++a){
            c2[i][a] = b2[i][a];
        }
    }

    Tensor Aov = A.block("ov");
    double diff_ov = difference(Aov, c2).second;

    return diff_oo + diff_ov;
}

double test_Cip_times_equal_double()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"oo","vv","ov","vo"});

    size_t no = 3;
    size_t nv = 5;

    Tensor Aoo_t = build_and_fill("Aoo", {no, no}, a2);
    Tensor Aov_t = build_and_fill("Aov", {no, nv}, b2);

    A.block("oo")("pq") = Aoo_t("pq");
    A.block("ov")("pq") = Aov_t("pq");

    A("ip") *= std::exp(1.0);

    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            c2[i][j] = std::exp(1.0) * a2[i][j];
        }
    }

    Tensor Aoo = A.block("oo");
    double diff_oo = difference(Aoo, c2).second;

    for (size_t i = 0; i < no; ++i){
        for (size_t a = 0; a < nv; ++a){
            c2[i][a] = std::exp(1.0) * b2[i][a];
        }
    }

    Tensor Aov = A.block("ov");
    double diff_ov = difference(Aov, c2).second;

    return diff_oo + diff_ov;
}

double test_Cij_equal_Aik_B_jk()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"oo","ov","vo","vv"});
    BlockedTensor B = BlockedTensor::build(kCore,"B",{"oo","ov","vo","vv"});
    BlockedTensor C = BlockedTensor::build(kCore,"C",{"oo","ov","vo","vv"});

    size_t no = 3;
    size_t nv = 5;

    Tensor Aoo_t = build_and_fill("Aoo", {no, no}, a2);
    Tensor Boo_t = build_and_fill("Boo", {no, no}, b2);
    Tensor Coo_t = build_and_fill("Coo", {no, no}, c2);

    A.block("oo")("pq") = Aoo_t("pq");
    B.block("oo")("pq") = Boo_t("pq");
    C.block("oo")("pq") = Coo_t("pq");


    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < no; ++k){
                c2[i][j] += a2[i][k] * b2[j][k];
            }
        }
    }

    C("ij") = A("ik") * B("jk");

    Tensor Coo = C.block("oo");
    double diff_oo = difference(Coo, c2).second;

    return diff_oo;
}

double test_Cij_equal_half_Aia_B_aj()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"oo","ov","vo","vv"});
    BlockedTensor B = BlockedTensor::build(kCore,"B",{"oo","ov","vo","vv"});
    BlockedTensor C = BlockedTensor::build(kCore,"C",{"oo","ov","vo","vv"});

    size_t no = 3;
    size_t nv = 5;

    Tensor Aov_t = build_and_fill("Aov", {no, nv}, a2);
    Tensor Bvo_t = build_and_fill("Bvo", {nv, no}, b2);
    Tensor Coo_t = build_and_fill("Coo", {no, no}, c2);

    A.block("ov")("pq") = Aov_t("pq");
    B.block("vo")("pq") = Bvo_t("pq");
    C.block("oo")("pq") = Coo_t("pq");


    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            c2[i][j] = 0.0;
            for (size_t a = 0; a < nv; ++a){
                c2[i][j] += 0.5 * a2[i][a] * b2[a][j];
            }
        }
    }

    C("ij") = 0.5 * A("ia") * B("aj");

    Tensor Coo = C.block("oo");
    double diff_oo = difference(Coo, c2).second;

    return diff_oo;
}

double test_Cij_plus_equal_half_Aai_B_ja()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"oo","ov","vo","vv"});
    BlockedTensor B = BlockedTensor::build(kCore,"B",{"oo","ov","vo","vv"});
    BlockedTensor C = BlockedTensor::build(kCore,"C",{"oo","ov","vo","vv"});

    size_t no = 3;
    size_t nv = 5;

    Tensor Avo_t = build_and_fill("Avo", {nv, no}, a2);
    Tensor Bov_t = build_and_fill("Bov", {no, nv}, b2);
    Tensor Coo_t = build_and_fill("Coo", {no, no}, c2);

    A.block("vo")("pq") = Avo_t("pq");
    B.block("ov")("pq") = Bov_t("pq");
    C.block("oo")("pq") = Coo_t("pq");

    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            for (size_t a = 0; a < nv; ++a){
                c2[i][j] += 0.5 * a2[a][i] * b2[j][a];
            }
        }
    }

    C("ij") += A("ai") * (0.5 * B("ja"));

    Tensor Coo = C.block("oo");
    double diff_oo = difference(Coo, c2).second;

    return diff_oo;
}

double test_Cij_minus_equal_Aik_B_jk()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"oo","ov","vo","vv"});
    BlockedTensor B = BlockedTensor::build(kCore,"B",{"oo","ov","vo","vv"});
    BlockedTensor C = BlockedTensor::build(kCore,"C",{"oo","ov","vo","vv"});

    size_t no = 3;
    size_t nv = 5;

    Tensor Aoo_t = build_and_fill("Aoo", {no, no}, a2);
    Tensor Boo_t = build_and_fill("Boo", {no, no}, b2);
    Tensor Coo_t = build_and_fill("Coo", {no, no}, c2);

    A.block("oo")("pq") = Aoo_t("pq");
    B.block("oo")("pq") = Boo_t("pq");
    C.block("oo")("pq") = Coo_t("pq");


    for (size_t i = 0; i < no; ++i){
        for (size_t j = 0; j < no; ++j){
            for (size_t k = 0; k < no; ++k){
                c2[i][j] -= a2[i][k] * b2[j][k];
            }
        }
    }

    C("ij") -= A("ik") * B("jk");

    Tensor Coo = C.block("oo");
    double diff_oo = difference(Coo, c2).second;

    return diff_oo;
}

double test_contraction_exception1()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k",{0,1,2},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});

    BlockedTensor A = BlockedTensor::build(kCore,"A",{"oo","ov","vo","vv"});
    BlockedTensor B = BlockedTensor::build(kCore,"B",{"oo","ov","vo","vv"});
    BlockedTensor C = BlockedTensor::build(kCore,"C",{"ov","vo","vv"});

    C("ij") = A("ia") * B("aj");

    return 0.0;
}

int main(int argc, char* argv[])
{
    printf(ANSI_COLOR_RESET);
    srand (time(nullptr));
    tensor::initialize(argc, argv);


    printf("==> Simple Operations <==\n\n");

    auto test_functions = {
            //            Expectation,  test function,  User friendly description
            std::make_tuple(kPass, test_mo_space, "Test"),
            std::make_tuple(kPass, test_add_mo_space, "Testing composite spaces"),
            std::make_tuple(kException, test_add_mo_space_nonexisting_space,"Testing addint nonexisting space"),
            std::make_tuple(kException, test_add_mo_space_repeated_index1,  "Testing adding repeated orbital indices (1)"),
            std::make_tuple(kException, test_add_mo_space_repeated_index2,  "Testing adding repeated orbital indices (2)"),
            std::make_tuple(kException, test_add_mo_space_repeated_index3,  "Testing adding repeated orbital indices (3)"),
            std::make_tuple(kException, test_add_mo_space_repeated_space1,  "Testing adding repeated orbital spaces (1)"),
            std::make_tuple(kException, test_add_mo_space_repeated_space2,  "Testing adding repeated orbital spaces (2)"),
            std::make_tuple(kException, test_add_mo_space_no_name1,         "Testing adding orbital space with no name (1)"),
            std::make_tuple(kException, test_add_mo_space_no_name2,         "Testing adding orbital space with no name (2)"),
            std::make_tuple(kException, test_add_mo_space_no_index1,        "Testing adding orbital space with no indices (1)"),
            std::make_tuple(kException, test_add_mo_space_no_index2,        "Testing adding orbital space with no indices (2)"),
            std::make_tuple(kPass,      test_add_mo_space_no_mos,           "Testing adding orbital space with no orbital list"),
            std::make_tuple(kPass,      test_block_creation1,               "Testing blocked tensor creation (1)"),
            std::make_tuple(kPass,      test_block_creation2,               "Testing blocked tensor creation (2)"),
            std::make_tuple(kPass,      test_block_creation3,               "Testing blocked tensor creation (3)"),
            std::make_tuple(kException, test_block_creation_bad_rank,       "Testing blocked tensor creation with variable rank"),
            std::make_tuple(kPass,      test_block_norm_1,                  "Testing blocked tensor 1-norm"),
            std::make_tuple(kPass,      test_block_norm_2,                  "Testing blocked tensor 2-norm"),
            std::make_tuple(kPass,      test_block_norm_3,                  "Testing blocked tensor inf-norm"),
            std::make_tuple(kPass,      test_block_zero,                    "Testing blocked tensor zero"),
            std::make_tuple(kPass,      test_block_scale,                   "Testing blocked tensor scale"),
            std::make_tuple(kPass,      test_block_labels1,                 "Testing blocked tensor labeling (1)"),
            std::make_tuple(kPass,      test_block_retrive_block1,          "Testing blocked tensor retrieve existing block"),
            std::make_tuple(kException, test_block_retrive_block2,          "Testing blocked tensor retrieve ambiguous block"),
            std::make_tuple(kException, test_block_retrive_block3,          "Testing blocked tensor retrieve null block (1)"),
            std::make_tuple(kException, test_block_retrive_block4,          "Testing blocked tensor retrieve null block (2)"),
            std::make_tuple(kPass,      test_Cij_equal_Aji,                 "Testing blocked tensor C(\"ij\") = A(\"ji\")"),
            std::make_tuple(kPass,      test_Cijab_plus_equal_Aaibj,        "Testing blocked tensor C(\"ijab\") += A(\"aibj\")"),
            std::make_tuple(kPass,      test_Cbija_minus_equal_Ajabi,       "Testing blocked tensor C(\"bija\") -= A(\"jabi\")"),
            std::make_tuple(kPass,      test_Cij_times_equal_double,        "Testing blocked tensor A(\"ij\") *= double"),
            std::make_tuple(kPass,      test_Cip_times_equal_double,        "Testing blocked tensor A(\"ip\") *= double"),
            std::make_tuple(kPass,      test_Cij_equal_Aik_B_jk,            "Testing blocked tensor C(\"ij\") = A(\"ik\") * B(\"jk\")"),
            std::make_tuple(kPass,      test_Cij_equal_half_Aia_B_aj,       "Testing blocked tensor C(\"ij\") = 0.5 * A(\"ia\") * B(\"aj\")"),
            std::make_tuple(kPass,      test_Cij_plus_equal_half_Aai_B_ja,  "Testing blocked tensor C(\"ij\") += A(\"ai\") * (0.5 * B(\"ja\"))"),
            std::make_tuple(kPass,      test_Cij_minus_equal_Aik_B_jk,      "Testing blocked tensor C(\"ij\") -= A(\"ik\") * B(\"jk\")"),
            std::make_tuple(kException, test_contraction_exception1,        "Testing blocked tensor contraction exception (1)"),
    };

    std::vector<std::tuple<std::string,TestResult,double>> results;

    printf(ANSI_COLOR_RESET);

    printf("\n %-60s %12s %s","Description","Max. error","Result");
    printf("\n %s",std::string(83,'-').c_str());

    bool success = true;
    for (auto test_function : test_functions) {
        printf("\n %-60s", std::get<2>(test_function));
        double result = 0.0;
        TestResult tresult = kPass, report_result = kPass;
        std::string exception;
        try {
            result = std::get<1>(test_function)();

            // Did the test pass based on returned value?
            tresult = std::fabs(result) < epsilon ? kPass : kFail;
            // Was the tresult the expected result? If so color green else red.
            report_result = tresult == std::get<0>(test_function) ? kPass : kFail;
        }
        catch (std::exception& e) {
            // was an exception expected?
            tresult = kException;
            report_result = tresult == std::get<0>(test_function) ? kPass : kException;

//            printf("\n  %s",e.what());
            if (report_result == kException) {
                exception = e.what();
            }
        }
        printf(" %7e", result);
        switch (report_result) {
            case kPass:
                printf(ANSI_COLOR_GREEN);
                break;
            case kFail:
                printf(ANSI_COLOR_RED);
                break;
            default:
                printf(ANSI_COLOR_YELLOW);
        }
        switch (tresult) {
            case kPass:
                printf(" Passed" ANSI_COLOR_RESET);
                break;
            case kFail:
                printf(" Failed" ANSI_COLOR_RESET);
                break;
            default:
                printf(" Exception" ANSI_COLOR_RESET);
        }

        if (report_result == kException)
            printf("\n    Unexpected: %s", exception.c_str());
        if (report_result != kPass)
            success = false;
    }
    printf("\n %s",std::string(83,'-').c_str());
    printf("\n Tests: %s\n",success ? "All passed" : "Some failed");

//    printf("%s\n",std::string(82,'-').c_str());
//    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed", "Delta");
//    printf("%s\n",std::string(82,'-').c_str());
//    success = true;
//    success &= test_function(try_relative_difference, "Relative Difference", kExact);
//    success &= test_function(try_1_norm             , "1-Norm"             , kEpsilon);
//    success &= test_function(try_2_norm             , "2-Norm"             , kEpsilon);
//    success &= test_function(try_inf_norm           , "Inf-Norm"           , kEpsilon);
//    success &= test_function(try_zero               , "Zero"               , kExact);
//    success &= test_function(try_copy               , "Copy"               , kExact);
//    success &= test_function(try_scale              , "Scale"              , kExact);
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Tests: %s\n\n",success ? "All passed" : "Some failed");

//    printf("==> Slice Operations <==\n\n");
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed", "Delta");
//    printf("%s\n",std::string(82,'-').c_str());
//    success = true;
//    success &= test_function(try_slice_rank0        , "Full Slice Rank-0"  , kExact);
//    success &= test_function(try_slice_rank3        , "Full Slice Rank-3"  , kExact);
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Tests: %s\n\n",success ? "All passed" : "Some failed");

//    printf("==> Permute Operations <==\n\n");
//    success = true;
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed", "Delta");
//    mode = 0; alpha = 1.0; beta = 0.0;
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
//    printf("%s\n",std::string(82,'-').c_str());
//    success &= test_function(try_permute_rank0      , "Permute Rank-0"      , kExact);
//    success &= test_function(try_permute_rank1_i    , "Permute Rank-1 i"    , kExact);
//    success &= test_function(try_permute_rank2_ij   , "Permute Rank-2 ij"   , kExact);
//    success &= test_function(try_permute_rank2_ji   , "Permute Rank-2 ji"   , kExact);
//    success &= test_function(try_permute_rank3_ijk  , "Permute Rank-3 ijk"  , kExact);
//    success &= test_function(try_permute_rank3_ikj  , "Permute Rank-3 ikj"  , kExact);
//    success &= test_function(try_permute_rank3_jik  , "Permute Rank-3 jik"  , kExact);
//    success &= test_function(try_permute_rank3_jki  , "Permute Rank-3 jki"  , kExact);
//    success &= test_function(try_permute_rank3_kij  , "Permute Rank-3 kij"  , kExact);
//    success &= test_function(try_permute_rank3_kji  , "Permute Rank-3 kji"  , kExact);
//    success &= test_function(try_permute_rank4_ijkl , "Permute Rank-4 ijkl" , kExact);
//    success &= test_function(try_permute_rank4_lkji , "Permute Rank-4 lkji" , kExact);
//    success &= test_function(try_permute_rank4_ijlk , "Permute Rank-4 ijlk" , kExact);
//    success &= test_function(try_permute_rank4_jikl , "Permute Rank-4 jikl" , kExact);
//    success &= test_function(try_permute_rank4_ikjl , "Permute Rank-4 ikjl" , kExact);
//    success &= test_function(try_permute_rank4_lkji , "Permute Rank-4 lkji" , kExact);
//    mode = 0; alpha = random_double(); beta = random_double();
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
//    printf("%s\n",std::string(82,'-').c_str());
//    success &= test_function(try_permute_rank0      , "Permute Rank-0"      , kExact);
//    success &= test_function(try_permute_rank1_i    , "Permute Rank-1 i"    , kExact);
//    success &= test_function(try_permute_rank2_ij   , "Permute Rank-2 ij"   , kExact);
//    success &= test_function(try_permute_rank2_ji   , "Permute Rank-2 ji"   , kExact);
//    success &= test_function(try_permute_rank3_ijk  , "Permute Rank-3 ijk"  , kExact);
//    success &= test_function(try_permute_rank3_ikj  , "Permute Rank-3 ikj"  , kExact);
//    success &= test_function(try_permute_rank3_jik  , "Permute Rank-3 jik"  , kExact);
//    success &= test_function(try_permute_rank3_jki  , "Permute Rank-3 jki"  , kExact);
//    success &= test_function(try_permute_rank3_kij  , "Permute Rank-3 kij"  , kExact);
//    success &= test_function(try_permute_rank3_kji  , "Permute Rank-3 kji"  , kExact);
//    success &= test_function(try_permute_rank4_ijkl , "Permute Rank-4 ijkl" , kExact);
//    success &= test_function(try_permute_rank4_ijlk , "Permute Rank-4 ijlk" , kExact);
//    success &= test_function(try_permute_rank4_jikl , "Permute Rank-4 jikl" , kExact);
//    success &= test_function(try_permute_rank4_ikjl , "Permute Rank-4 ikjl" , kExact);
//    success &= test_function(try_permute_rank4_lkji , "Permute Rank-4 lkji" , kExact);
//    mode = 1; alpha = 1.0; beta = 0.0;
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Operator Overloading: =\n");
//    printf("%s\n",std::string(82,'-').c_str());
//    success &= test_function(try_permute_rank0      , "Permute Rank-0"      , kExact);
//    success &= test_function(try_permute_rank1_i    , "Permute Rank-1 i"    , kExact);
//    success &= test_function(try_permute_rank2_ij   , "Permute Rank-2 ij"   , kExact);
//    success &= test_function(try_permute_rank2_ji   , "Permute Rank-2 ji"   , kExact);
//    success &= test_function(try_permute_rank3_ijk  , "Permute Rank-3 ijk"  , kExact);
//    success &= test_function(try_permute_rank3_ikj  , "Permute Rank-3 ikj"  , kExact);
//    success &= test_function(try_permute_rank3_jik  , "Permute Rank-3 jik"  , kExact);
//    success &= test_function(try_permute_rank3_jki  , "Permute Rank-3 jki"  , kExact);
//    success &= test_function(try_permute_rank3_kij  , "Permute Rank-3 kij"  , kExact);
//    success &= test_function(try_permute_rank3_kji  , "Permute Rank-3 kji"  , kExact);
//    success &= test_function(try_permute_rank4_ijkl , "Permute Rank-4 ijkl" , kExact);
//    success &= test_function(try_permute_rank4_ijlk , "Permute Rank-4 ijlk" , kExact);
//    success &= test_function(try_permute_rank4_jikl , "Permute Rank-4 jikl" , kExact);
//    success &= test_function(try_permute_rank4_ikjl , "Permute Rank-4 ikjl" , kExact);
//    success &= test_function(try_permute_rank4_lkji , "Permute Rank-4 lkji" , kExact);
//    mode = 2; alpha = 1.0; beta = 1.0;
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Operator Overloading: +=\n");
//    printf("%s\n",std::string(82,'-').c_str());
//    success &= test_function(try_permute_rank0      , "Permute Rank-0"      , kExact);
//    success &= test_function(try_permute_rank1_i    , "Permute Rank-1 i"    , kExact);
//    success &= test_function(try_permute_rank2_ij   , "Permute Rank-2 ij"   , kExact);
//    success &= test_function(try_permute_rank2_ji   , "Permute Rank-2 ji"   , kExact);
//    success &= test_function(try_permute_rank3_ijk  , "Permute Rank-3 ijk"  , kExact);
//    success &= test_function(try_permute_rank3_ikj  , "Permute Rank-3 ikj"  , kExact);
//    success &= test_function(try_permute_rank3_jik  , "Permute Rank-3 jik"  , kExact);
//    success &= test_function(try_permute_rank3_jki  , "Permute Rank-3 jki"  , kExact);
//    success &= test_function(try_permute_rank3_kij  , "Permute Rank-3 kij"  , kExact);
//    success &= test_function(try_permute_rank3_kji  , "Permute Rank-3 kji"  , kExact);
//    success &= test_function(try_permute_rank4_ijkl , "Permute Rank-4 ijkl" , kExact);
//    success &= test_function(try_permute_rank4_ijlk , "Permute Rank-4 ijlk" , kExact);
//    success &= test_function(try_permute_rank4_jikl , "Permute Rank-4 jikl" , kExact);
//    success &= test_function(try_permute_rank4_ikjl , "Permute Rank-4 ikjl" , kExact);
//    success &= test_function(try_permute_rank4_lkji , "Permute Rank-4 lkji" , kExact);
//    mode = 3; alpha = -1.0; beta = 1.0;
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Operator Overloading: -=\n");
//    printf("%s\n",std::string(82,'-').c_str());
//    success &= test_function(try_permute_rank0      , "Permute Rank-0"      , kExact);
//    success &= test_function(try_permute_rank1_i    , "Permute Rank-1 i"    , kExact);
//    success &= test_function(try_permute_rank2_ij   , "Permute Rank-2 ij"   , kExact);
//    success &= test_function(try_permute_rank2_ji   , "Permute Rank-2 ji"   , kExact);
//    success &= test_function(try_permute_rank3_ijk  , "Permute Rank-3 ijk"  , kExact);
//    success &= test_function(try_permute_rank3_ikj  , "Permute Rank-3 ikj"  , kExact);
//    success &= test_function(try_permute_rank3_jik  , "Permute Rank-3 jik"  , kExact);
//    success &= test_function(try_permute_rank3_jki  , "Permute Rank-3 jki"  , kExact);
//    success &= test_function(try_permute_rank3_kij  , "Permute Rank-3 kij"  , kExact);
//    success &= test_function(try_permute_rank3_kji  , "Permute Rank-3 kji"  , kExact);
//    success &= test_function(try_permute_rank4_ijkl , "Permute Rank-4 ijkl" , kExact);
//    success &= test_function(try_permute_rank4_ijlk , "Permute Rank-4 ijlk" , kExact);
//    success &= test_function(try_permute_rank4_jikl , "Permute Rank-4 jikl" , kExact);
//    success &= test_function(try_permute_rank4_ikjl , "Permute Rank-4 ikjl" , kExact);
//    success &= test_function(try_permute_rank4_lkji , "Permute Rank-4 lkji" , kExact);
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Tests: %s\n\n",success ? "All Passed" : "Some Failed");

//    printf("==> Permute Exceptions <==\n\n");
//    success = true;
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed", "Delta");
//    mode = 0; alpha = 1.0; beta = 0.0;
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
//    printf("%s\n",std::string(82,'-').c_str());
//    success &= test_function(try_permute_label_fail  , "Permute Label Fail"   , kException);
//    success &= test_function(try_permute_rank_fail   , "Permute Rank Fail"    , kException);
//    success &= test_function(try_permute_index_fail  , "Permute Index Fail"   , kException);
//    success &= test_function(try_permute_size_fail   , "Permute Size Fail"    , kException);
//    printf("%s\n",std::string(82,'-').c_str());
//    printf("Tests: %s\n\n",success ? "All Passed" : "Some Failed");


    tensor::finalize();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
