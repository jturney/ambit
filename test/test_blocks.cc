#include <tensor/blocked_tensor.h>
#include <cmath>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

using namespace tensor;

/// Expected relative accuracy for numerical exactness
const double epsilon = 1.0E-14;

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


double test_mo_space()
{
    MOSpace alpha_occ("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    MOSpace alpha_vir("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
//    alpha_occ.print();
//    alpha_vir.print();
    return 0.0;
}

double test_add_mo_space()
{
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::add_mo_space("o","i,j,k,l",{0,1,2,3,4},AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d",{5,6,7,8,9},AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s,t",{"o","v"});
//    BlockedTensor::print_mo_spaces();
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
    };

    std::vector<std::tuple<std::string,TestResult,double>> results;

    printf(ANSI_COLOR_RESET);

    printf("\n %-50s %12s %s","Description","Max. error","Result");
    printf("\n %s",std::string(73,'-').c_str());

    bool success = true;
    for (auto test_function : test_functions) {
        printf("\n %-50s", std::get<2>(test_function));
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
    printf("\n %s",std::string(73,'-').c_str());
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
