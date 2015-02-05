#include <tensor/tensor.h>
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
    kExact,     // Some op occurred and the results match exactly
    kEpsilon,   // Some op occurred and the results match to something proportional to epsilon
    kDeviation, // Some op occurred and the results deviate significantly from epsilon
    kException  // Some op occurred and the op threw an exception
};

std::string test_result_string(TestResult type)
{
    switch (type) {
        case kExact:
            return "Exact";
            break;
        case kEpsilon:
            return "Epsilon";
            break;
        case kDeviation:
            return "Deviation";
            break;
        case kException:
            return "Exception";
            break;
        default:
            return "Unknown";
    }
}

/// Initialize A1 and A2 to the same random fill
void initialize_random(Tensor& A1, Tensor& A2)
{
    size_t numel1 = A1.numel();
    size_t numel2 = A2.numel();
    if (numel1 != numel2) throw std::runtime_error("Tensors do not have same numel.");
    std::vector<double>& A1v = A1.data();
    std::vector<double>& A2v = A2.data();
    for (size_t ind = 0L; ind < numel1; ind++) {
        double randnum = double(std::rand())/double(RAND_MAX);
        A1v[ind] = randnum;
        A2v[ind] = randnum;
    }
}

/// Returns |A1 - A2|_INF / |A2|_INF or 0.0 if |A1 - A2|_INF == 0.0
double relative_difference(const Tensor& A1, const Tensor& A2) 
{
    size_t numel1 = A1.numel();
    size_t numel2 = A2.numel();
    if (numel1 != numel2) throw std::runtime_error("Tensors do not have same numel.");
    const std::vector<double>& A1v = A1.data();
    const std::vector<double>& A2v = A2.data();
    double dmax = 0.0;
    double Dmax = 0.0;
    for (size_t ind = 0L; ind < numel1; ind++) {
        double d = fabs(A1v[ind] - A2v[ind]);
        double D = fabs(A2v[ind]);
        dmax = std::max(d,dmax);
        Dmax = std::max(D,Dmax);
    }
    if (dmax == 0.0) return 0.0;
    else return dmax / Dmax;
}

bool test_relative_difference()
{
    // > USER CODE < //
    TestResult expected = kExact;
    std::string label = "Relative Difference"; 
    // > END USER CODE < //

    TestResult observed = kDeviation;
    double delta = 0.0;
    std::string exception;

    try {
        // > USER CODE < //
        Dimension Adims = {4,5,6};
        Tensor A1 = Tensor::build(kCore, "A1", Adims);
        Tensor A2 = Tensor::build(kCore, "A2", Adims);
        initialize_random(A1, A2);
        delta = relative_difference(A1,A2);
        // > END USER CODE < //

        if (delta == 0.0) observed = kExact;
        else if (delta <= epsilon) observed = kEpsilon;
        else observed = kDeviation;
    } catch (std::exception& e) {
        observed = kException;
        if (expected != kException) exception = e.what(); 
    }  

    bool pass = observed == expected;
    if (pass) printf(ANSI_COLOR_GREEN);
    else printf(ANSI_COLOR_RED);

    printf(" %-50s ", label.c_str());
    printf(" %-9s ", test_result_string(expected).c_str()); 
    printf(" %-9s ", test_result_string(observed).c_str()); 
    printf(" %11.3E\n", delta); 

    printf(ANSI_COLOR_RESET);

    if (exception.size())
        printf(" Exception: %s\n", exception.c_str()); 

    return pass;
}

bool test_zero()
{
    // > USER CODE < //
    TestResult expected = kExact;
    std::string label = "Zero"; 
    // > END USER CODE < //

    TestResult observed = kDeviation;
    double delta = 0.0;
    std::string exception;

    try {
        // > USER CODE < //
        Dimension Adims = {4,5,6};
        Tensor A1 = Tensor::build(kCore, "A1", Adims);
        Tensor A2 = Tensor::build(kCore, "A2", Adims);
        initialize_random(A1, A2);

        A1.zero();
        
        size_t numel = A2.numel();
        std::vector<double>& A2v = A2.data();
        for (size_t ind = 0L; ind < numel; ind++) {
            A2v[ind] = 0.0;
        }

        delta = relative_difference(A1,A2);
        // > END USER CODE < //

        if (delta == 0.0) observed = kExact;
        else if (delta <= epsilon) observed = kEpsilon;
        else observed = kDeviation;
    } catch (std::exception& e) {
        observed = kException;
        if (expected != kException) exception = e.what(); 
    }  

    bool pass = observed == expected;
    if (pass) printf(ANSI_COLOR_GREEN);
    else printf(ANSI_COLOR_RED);

    printf(" %-50s ", label.c_str());
    printf(" %-9s ", test_result_string(expected).c_str()); 
    printf(" %-9s ", test_result_string(observed).c_str()); 
    printf(" %11.3E\n", delta); 

    printf(ANSI_COLOR_RESET);

    if (exception.size())
        printf(" Exception: %s\n", exception.c_str()); 

    return pass;
}

bool test_copy()
{
    // > USER CODE < //
    TestResult expected = kExact;
    std::string label = "Copy"; 
    // > END USER CODE < //

    TestResult observed = kDeviation;
    double delta = 0.0;
    std::string exception;

    try {
        // > USER CODE < //
        Dimension Adims = {4,5,6};
        Tensor A1 = Tensor::build(kCore, "A1", Adims);
        Tensor A2 = Tensor::build(kCore, "A2", Adims);
        initialize_random(A1, A2);

        A1.zero();
        A1.copy(A2);
        
        delta = relative_difference(A1,A2);
        // > END USER CODE < //

        if (delta == 0.0) observed = kExact;
        else if (delta <= epsilon) observed = kEpsilon;
        else observed = kDeviation;
    } catch (std::exception& e) {
        observed = kException;
        if (expected != kException) exception = e.what(); 
    }  

    bool pass = observed == expected;
    if (pass) printf(ANSI_COLOR_GREEN);
    else printf(ANSI_COLOR_RED);

    printf(" %-50s ", label.c_str());
    printf(" %-9s ", test_result_string(expected).c_str()); 
    printf(" %-9s ", test_result_string(observed).c_str()); 
    printf(" %11.3E\n", delta); 

    printf(ANSI_COLOR_RESET);

    if (exception.size())
        printf(" Exception: %s\n", exception.c_str()); 

    return pass;
}

bool test_scale()
{
    // > USER CODE < //
    TestResult expected = kExact;
    std::string label = "Scale"; 
    // > END USER CODE < //

    TestResult observed = kDeviation;
    double delta = 0.0;
    std::string exception;

    try {
        // > USER CODE < //
        Dimension Adims = {4,5,6};
        Tensor A1 = Tensor::build(kCore, "A1", Adims);
        Tensor A2 = Tensor::build(kCore, "A2", Adims);
        initialize_random(A1, A2);

        A1.scale(M_PI);
        
        size_t numel = A2.numel();
        std::vector<double>& A2v = A2.data();
        for (size_t ind = 0L; ind < numel; ind++) {
            A2v[ind] *= M_PI;
        }

        delta = relative_difference(A1,A2);
        // > END USER CODE < //

        if (delta == 0.0) observed = kExact;
        else if (delta <= epsilon) observed = kEpsilon;
        else observed = kDeviation;
    } catch (std::exception& e) {
        observed = kException;
        if (expected != kException) exception = e.what(); 
    }  

    bool pass = observed == expected;
    if (pass) printf(ANSI_COLOR_GREEN);
    else printf(ANSI_COLOR_RED);

    printf(" %-50s ", label.c_str());
    printf(" %-9s ", test_result_string(expected).c_str()); 
    printf(" %-9s ", test_result_string(observed).c_str()); 
    printf(" %11.3E\n", delta); 

    printf(ANSI_COLOR_RESET);

    if (exception.size())
        printf(" Exception: %s\n", exception.c_str()); 

    return pass;
}



int main(int argc, char* argv[])
{
    printf(ANSI_COLOR_RESET);
    srand (time(NULL));
    tensor::initialize(argc, argv);

    bool success = true;

    success = success && test_relative_difference();
    success = success && test_zero();
    success = success && test_copy();
    success = success && test_scale();

    tensor::finalize();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

