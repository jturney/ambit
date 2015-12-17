#include <ambit/tensor.h>
#include <ambit/print.h>
#include <ambit/io/io.h>
#include <ambit/helpers/psi4/io.h>
#include <cmath>
#include <assert.h>
#include <cstdlib>

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"

using namespace ambit;

/// Expected relative accuracy for numerical exactness
const double epsilon = 1.0E-14;

/// Sc/haheme to categorize expected vs. actual op behavior
enum TestResult
{
    kExact,     // Some op occurred and the results match exactly
    kEpsilon,   // Some op occurred and the results match to something
                // proportional to epsilon
    kDeviation, // Some op occurred and the results deviate significantly from
                // epsilon
    kException  // Some op occurred and the op threw an exception
};

std::string test_result_string(TestResult type)
{
    switch (type)
    {
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

/// Try a function which returns a double against an expected behavior
bool test_function(double (*trial_function)(), const std::string &label,
                   TestResult expected)
{
    TestResult observed = kDeviation;
    double delta = 0.0;
    std::string exception;

    try
    {
        delta = trial_function();
        if (delta == 0.0 && expected != kEpsilon)
            observed = kExact;
        else if (delta <= epsilon)
            observed = kEpsilon;
        else
            observed = kDeviation;
    }
    catch (std::exception &e)
    {
        observed = kException;
        exception = e.what();
    }

    bool pass = observed == expected;
    if (pass)
        printf(ANSI_COLOR_GREEN);
    else
        printf(ANSI_COLOR_RED);

    printf("%-50s ", label.c_str());
    printf("%-9s ", test_result_string(expected).c_str());
    printf("%-9s ", test_result_string(observed).c_str());
    printf("%11.3E\n", delta);

    printf(ANSI_COLOR_RESET);

    if (exception.size())
        printf(" Exception: %s\n", exception.c_str());

    return pass;
}

double random_double() { return double(std::rand()) / double(RAND_MAX); }

/// Initialize A1 to some random fill
void initialize_random(Tensor &A1)
{
    size_t numel1 = A1.numel();
    std::vector<double> &A1v = A1.data();
    for (size_t ind = 0L; ind < numel1; ind++)
    {
        double randnum = double(std::rand()) / double(RAND_MAX);
        A1v[ind] = randnum;
    }
}

/// Initialize A1 and A2 to the same random fill
void initialize_random(Tensor &A1, Tensor &A2)
{
    size_t numel1 = A1.numel();
    size_t numel2 = A2.numel();
    if (numel1 != numel2)
        throw std::runtime_error("Tensors do not have same numel.");
    std::vector<double> &A1v = A1.data();
    std::vector<double> &A2v = A2.data();
    for (size_t ind = 0L; ind < numel1; ind++)
    {
        double randnum = double(std::rand()) / double(RAND_MAX);
        A1v[ind] = randnum;
        A2v[ind] = randnum;
    }
}

/// Returns |A1 - A2|_INF / |A2|_INF or 0.0 if |A1 - A2|_INF == 0.0
double relative_difference(const Tensor &A1, const Tensor &A2)
{
    size_t numel1 = A1.numel();
    size_t numel2 = A2.numel();
    if (numel1 != numel2)
        throw std::runtime_error("Tensors do not have same numel.");
    const std::vector<double> &A1v = A1.data();
    const std::vector<double> &A2v = A2.data();
    double dmax = 0.0;
    double Dmax = 0.0;
    for (size_t ind = 0L; ind < numel1; ind++)
    {
        double d = fabs(A1v[ind] - A2v[ind]);
        double D = fabs(A2v[ind]);
        dmax = std::max(d, dmax);
        Dmax = std::max(D, Dmax);
    }
    if (dmax == 0.0)
        return 0.0;
    else
        return dmax / Dmax;
}

int main(int argc, char *argv[])
{
    printf(ANSI_COLOR_RESET);
    srand(time(nullptr));
    ambit::settings::debug = true;
    ambit::initialize(argc, argv);

    bool success = true;
    int nirrep, nso;

    {
        ambit::io::File file32("test.32", ambit::io::kOpenModeOpenExisting);

        file32.read("::Num. irreps", &nirrep, 1);
        print("nirrep = %d\n", nirrep);
        assert(nirrep == 1);

        file32.read("::Num. SO", &nso, 1);
        print("nso = %d\n", nso);
    }

    Dimension AO2 = {(size_t)nso, (size_t)nso};
    Dimension AO4 = {(size_t)nso, (size_t)nso, (size_t)nso, (size_t)nso};
    {
        Tensor S = Tensor::build(kDistributed, "Overlap", AO2);
        helpers::psi4::load_matrix("test.35", "SO-basis Overlap Ints", S);

        double norm = S.norm();
        printn("norm of S is %lf\n", settings::rank, norm);
    }

    {
        Tensor g = Tensor::build(kDistributed, "g", AO4);
        helpers::psi4::load_iwl("test.33", g);

        double norm = g.norm();
        printn("norm of g is %lf\n", settings::rank, norm);

        printn("max value in g is %lf\n", std::get<0>(g.max()));
        printn("min value in g is %lf\n", std::get<0>(g.min()));
    }

    ambit::finalize();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
