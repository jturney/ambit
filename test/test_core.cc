#include <ambit/tensor.h>
#include <cmath>
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

/// Scheme to categorize expected vs. actual op behavior
enum TestResult
{
    kExact,     // Some op occurred and the results match exactly
    kEpsilon,   // Some op occurred and the results match to something
                // proportional to epsilon
    kDeviation, // Some op occurred and the results deviate significantly from
                // epsilon
    kException  // Some op occurred and the op threw an exception
};

/// Global alpha parameter
double alpha = 1.0;
/// Global beta parameter
double beta = 0.0;
/// 0 - explicit call, 1 - = OO, 2 - += OO, 3 - -= OO
int mode = 0;

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

double try_relative_difference()
{
    Dimension Adims = {4, 5, 6};
    Tensor A1 = Tensor::build(CoreTensor, "A1", Adims);
    Tensor A2 = Tensor::build(CoreTensor, "A2", Adims);
    initialize_random(A1, A2);
    return relative_difference(A1, A2);
}

double try_1_norm()
{
    Dimension Adims = {4, 5, 6};
    Tensor A1 = Tensor::build(CoreTensor, "A1", Adims);
    Tensor A2 = Tensor::build(CoreTensor, "A2", Adims);
    initialize_random(A1, A2);

    double normA1 = A2.norm(1);

    double normA2 = 0.0;
    size_t numel = A2.numel();
    std::vector<double> &A2v = A2.data();
    for (size_t ind = 0L; ind < numel; ind++)
    {
        normA2 += fabs(A2v[ind]);
    }

    double delta = fabs(normA1 - normA2);
    if (delta == 0.0)
        return 0.0;
    else
        return delta / fabs(normA1 + normA2);
}

double try_2_norm()
{
    Dimension Adims = {4, 5, 6};
    Tensor A1 = Tensor::build(CoreTensor, "A1", Adims);
    Tensor A2 = Tensor::build(CoreTensor, "A2", Adims);
    initialize_random(A1, A2);

    double normA1 = A2.norm(2);

    double normA2 = 0.0;
    size_t numel = A2.numel();
    std::vector<double> &A2v = A2.data();
    for (size_t ind = 0L; ind < numel; ind++)
    {
        normA2 += A2v[ind] * A2v[ind];
    }
    normA2 = sqrt(normA2);

    double delta = fabs(normA1 - normA2);
    if (delta == 0.0)
        return 0.0;
    else
        return delta / fabs(normA1 + normA2);
}

double try_inf_norm()
{
    Dimension Adims = {4, 5, 6};
    Tensor A1 = Tensor::build(CoreTensor, "A1", Adims);
    Tensor A2 = Tensor::build(CoreTensor, "A2", Adims);
    initialize_random(A1, A2);

    double normA1 = A2.norm(0);

    double normA2 = 0.0;
    size_t numel = A2.numel();
    std::vector<double> &A2v = A2.data();
    for (size_t ind = 0L; ind < numel; ind++)
    {
        normA2 = std::max(normA2, A2v[ind]);
    }

    double delta = fabs(normA1 - normA2);
    if (delta == 0.0)
        return 0.0;
    else
        return delta / fabs(normA1 + normA2);
}

double try_zero()
{
    Dimension Adims = {4, 5, 6};
    Tensor A1 = Tensor::build(CoreTensor, "A1", Adims);
    Tensor A2 = Tensor::build(CoreTensor, "A2", Adims);
    initialize_random(A1, A2);

    A1.zero();

    size_t numel = A2.numel();
    std::vector<double> &A2v = A2.data();
    for (size_t ind = 0L; ind < numel; ind++)
    {
        A2v[ind] = 0.0;
    }

    return relative_difference(A1, A2);
}

double try_copy()
{
    Dimension Adims = {4, 5, 6};
    Tensor A1 = Tensor::build(CoreTensor, "A1", Adims);
    Tensor A2 = Tensor::build(CoreTensor, "A2", Adims);
    initialize_random(A1, A2);

    A1.zero();
    A1.copy(A2);

    return relative_difference(A1, A2);
}

double try_scale()
{
    Dimension Adims = {4, 5, 6};
    Tensor A1 = Tensor::build(CoreTensor, "A1", Adims);
    Tensor A2 = Tensor::build(CoreTensor, "A2", Adims);
    initialize_random(A1, A2);

    double s = random_double();
    A1.scale(s);

    size_t numel = A2.numel();
    std::vector<double> &A2v = A2.data();
    for (size_t ind = 0L; ind < numel; ind++)
    {
        A2v[ind] *= s;
    }

    return relative_difference(A1, A2);
}

double try_slice_rank0_same1()
{
    Dimension Cdims = {};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.slice(A, {}, {}, alpha, beta);
    else if (mode == 1)
        C1() = A();
    else if (mode == 2)
        C1() += A();
    else if (mode == 3)
        C1() -= A();
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    Cv[0] = alpha * Av[0] + beta * Cv[0];

    return relative_difference(C1, C2);
}

double try_slice_rank1_same1()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{0L, 4L}};
    IndexRange Ainds = {{0L, 4L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1() = A();
    else if (mode == 2)
        C1() += A();
    else if (mode == 3)
        C1() -= A();
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        Cv[(i + Cinds[0][0])] =
            alpha * Av[(i + Ainds[0][0])] + beta * Cv[(i + Cinds[0][0])];
    }

    return relative_difference(C1, C2);
}
double try_slice_rank1_same2()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 3L}};
    IndexRange Ainds = {{2L, 4L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        Cv[(i + Cinds[0][0])] =
            alpha * Av[(i + Ainds[0][0])] + beta * Cv[(i + Cinds[0][0])];
    }

    return relative_difference(C1, C2);
}
double try_slice_rank1_diff1()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 4L}};
    IndexRange Ainds = {{0L, 3L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        Cv[(i + Cinds[0][0])] =
            alpha * Av[(i + Ainds[0][0])] + beta * Cv[(i + Cinds[0][0])];
    }

    return relative_difference(C1, C2);
}
double try_slice_rank1_diff2()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{2L, 4L}};
    IndexRange Ainds = {{1L, 3L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        Cv[(i + Cinds[0][0])] =
            alpha * Av[(i + Ainds[0][0])] + beta * Cv[(i + Cinds[0][0])];
    }

    return relative_difference(C1, C2);
}

double try_slice_rank2_same1()
{
    Dimension Cdims = {4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{0L, 4L}, {0L, 5L}};
    IndexRange Ainds = {{0L, 4L}, {0L, 5L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1() = A();
    else if (mode == 2)
        C1() += A();
    else if (mode == 3)
        C1() -= A();
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])] =
                alpha * Av[(i + Ainds[0][0]) * Adims[1] + (j + Ainds[1][0])] +
                beta * Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])];
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank2_same2()
{
    Dimension Cdims = {4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 3L}, {0L, 5L}};
    IndexRange Ainds = {{2L, 4L}, {0L, 5L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])] =
                alpha * Av[(i + Ainds[0][0]) * Adims[1] + (j + Ainds[1][0])] +
                beta * Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])];
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank2_same3()
{
    Dimension Cdims = {4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 3L}, {2L, 5L}};
    IndexRange Ainds = {{2L, 4L}, {1L, 4L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])] =
                alpha * Av[(i + Ainds[0][0]) * Adims[1] + (j + Ainds[1][0])] +
                beta * Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])];
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank2_diff1()
{
    Dimension Cdims = {4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 4L}, {1L, 5L}};
    IndexRange Ainds = {{0L, 3L}, {0L, 4L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])] =
                alpha * Av[(i + Ainds[0][0]) * Adims[1] + (j + Ainds[1][0])] +
                beta * Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])];
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank2_diff2()
{
    Dimension Cdims = {4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{2L, 4L}, {1L, 5L}};
    IndexRange Ainds = {{1L, 3L}, {0L, 4L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])] =
                alpha * Av[(i + Ainds[0][0]) * Adims[1] + (j + Ainds[1][0])] +
                beta * Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])];
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank2_diff3()
{
    Dimension Cdims = {4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{2L, 4L}, {2L, 5L}};
    IndexRange Ainds = {{1L, 3L}, {1L, 4L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])] =
                alpha * Av[(i + Ainds[0][0]) * Adims[1] + (j + Ainds[1][0])] +
                beta * Cv[(i + Cinds[0][0]) * Cdims[1] + (j + Cinds[1][0])];
        }
    }

    return relative_difference(C1, C2);
}

double try_slice_rank3_same1()
{
    Dimension Cdims = {4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5, 6};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{0L, 4L}, {0L, 5L}, {0L, 6L}};
    IndexRange Ainds = {{0L, 4L}, {0L, 5L}, {0L, 6L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1() = A();
    else if (mode == 2)
        C1() += A();
    else if (mode == 3)
        C1() -= A();
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            for (size_t k = 0; k < Cinds[2][1] - Cinds[2][0]; k++)
            {
                Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                   (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])] =
                    alpha *
                        Av[(i + Ainds[0][0]) * Adims[1] * Adims[2] +
                           (j + Ainds[1][0]) * Adims[2] + (k + Ainds[2][0])] +
                    beta * Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                              (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])];
            }
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank3_same2()
{
    Dimension Cdims = {4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5, 6};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 3L}, {0L, 5L}, {0L, 6L}};
    IndexRange Ainds = {{2L, 4L}, {0L, 5L}, {0L, 6L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            for (size_t k = 0; k < Cinds[2][1] - Cinds[2][0]; k++)
            {
                Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                   (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])] =
                    alpha *
                        Av[(i + Ainds[0][0]) * Adims[1] * Adims[2] +
                           (j + Ainds[1][0]) * Adims[2] + (k + Ainds[2][0])] +
                    beta * Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                              (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])];
            }
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank3_same3()
{
    Dimension Cdims = {4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5, 6};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 3L}, {2L, 5L}, {0L, 6L}};
    IndexRange Ainds = {{2L, 4L}, {1L, 4L}, {0L, 6L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            for (size_t k = 0; k < Cinds[2][1] - Cinds[2][0]; k++)
            {
                Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                   (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])] =
                    alpha *
                        Av[(i + Ainds[0][0]) * Adims[1] * Adims[2] +
                           (j + Ainds[1][0]) * Adims[2] + (k + Ainds[2][0])] +
                    beta * Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                              (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])];
            }
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank3_same4()
{
    Dimension Cdims = {4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5, 6};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 3L}, {2L, 5L}, {2L, 4L}};
    IndexRange Ainds = {{2L, 4L}, {1L, 4L}, {3L, 5L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            for (size_t k = 0; k < Cinds[2][1] - Cinds[2][0]; k++)
            {
                Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                   (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])] =
                    alpha *
                        Av[(i + Ainds[0][0]) * Adims[1] * Adims[2] +
                           (j + Ainds[1][0]) * Adims[2] + (k + Ainds[2][0])] +
                    beta * Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                              (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])];
            }
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank3_diff1()
{
    Dimension Cdims = {4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{1L, 4L}, {1L, 5L}, {1L, 6L}};
    IndexRange Ainds = {{0L, 3L}, {0L, 4L}, {0L, 5L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            for (size_t k = 0; k < Cinds[2][1] - Cinds[2][0]; k++)
            {
                Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                   (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])] =
                    alpha *
                        Av[(i + Ainds[0][0]) * Adims[1] * Adims[2] +
                           (j + Ainds[1][0]) * Adims[2] + (k + Ainds[2][0])] +
                    beta * Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                              (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])];
            }
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank3_diff2()
{
    Dimension Cdims = {4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{2L, 4L}, {1L, 5L}, {1L, 6L}};
    IndexRange Ainds = {{1L, 3L}, {0L, 4L}, {0L, 5L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            for (size_t k = 0; k < Cinds[2][1] - Cinds[2][0]; k++)
            {
                Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                   (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])] =
                    alpha *
                        Av[(i + Ainds[0][0]) * Adims[1] * Adims[2] +
                           (j + Ainds[1][0]) * Adims[2] + (k + Ainds[2][0])] +
                    beta * Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                              (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])];
            }
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank3_diff3()
{
    Dimension Cdims = {4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{2L, 4L}, {2L, 5L}, {1L, 6L}};
    IndexRange Ainds = {{1L, 3L}, {1L, 4L}, {0L, 5L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            for (size_t k = 0; k < Cinds[2][1] - Cinds[2][0]; k++)
            {
                Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                   (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])] =
                    alpha *
                        Av[(i + Ainds[0][0]) * Adims[1] * Adims[2] +
                           (j + Ainds[1][0]) * Adims[2] + (k + Ainds[2][0])] +
                    beta * Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                              (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])];
            }
        }
    }

    return relative_difference(C1, C2);
}
double try_slice_rank3_diff4()
{
    Dimension Cdims = {4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);

    IndexRange Cinds = {{2L, 4L}, {2L, 5L}, {2L, 6L}};
    IndexRange Ainds = {{1L, 3L}, {1L, 4L}, {1L, 5L}};

    if (mode == 0)
        C1.slice(A, Cinds, Ainds, alpha, beta);
    else if (mode == 1)
        C1(Cinds) = A(Ainds);
    else if (mode == 2)
        C1(Cinds) += A(Ainds);
    else if (mode == 3)
        C1(Cinds) -= A(Ainds);
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++)
    {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++)
        {
            for (size_t k = 0; k < Cinds[2][1] - Cinds[2][0]; k++)
            {
                Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                   (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])] =
                    alpha *
                        Av[(i + Ainds[0][0]) * Adims[1] * Adims[2] +
                           (j + Ainds[1][0]) * Adims[2] + (k + Ainds[2][0])] +
                    beta * Cv[(i + Cinds[0][0]) * Cdims[1] * Cdims[2] +
                              (j + Cinds[1][0]) * Cdims[2] + (k + Cinds[2][0])];
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_slice_label_fail()
{
    Tensor C1 = Tensor::build(CoreTensor, "C1", {4, 5});
    Tensor C2 = Tensor::build(CoreTensor, "C2", {4, 5});

    C1((IndexRange){{0L, 4L}}) = C2();
    return 0.0;
}
double try_slice_rank_fail()
{
    Tensor C1 = Tensor::build(CoreTensor, "C1", {4, 5});
    Tensor C2 = Tensor::build(CoreTensor, "C2", {4});

    C1() = C2();
    return 0.0;
}
double try_slice_size_fail()
{
    Tensor C1 = Tensor::build(CoreTensor, "C1", {4, 5});
    Tensor C2 = Tensor::build(CoreTensor, "C2", {4, 5});

    C1({{0, 4}, {0, 4}}) = C2();
    return 0.0;
}
double try_slice_bounds_fail()
{
    Tensor C1 = Tensor::build(CoreTensor, "C1", {4, 5});
    Tensor C2 = Tensor::build(CoreTensor, "C2", {4, 5});

    C1({{0, 6}, {0, 5}}) = C2({{0, 6}, {0, 5}});
    return 0.0;
}

double try_permute_rank0()
{
    Dimension Cdims = {};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {}, {}, alpha, beta);
    else if (mode == 1)
        C1("") = A("");
    else if (mode == 2)
        C1("") += A("");
    else if (mode == 3)
        C1("") -= A("");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    Cv[0] = alpha * Av[0] + beta * Cv[0];

    return relative_difference(C1, C2);
}

double try_permute_rank1_i()
{
    Dimension Cdims = {3};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i"}, {"i"}, alpha, beta);
    else if (mode == 1)
        C1("i") = A("i");
    else if (mode == 2)
        C1("i") += A("i");
    else if (mode == 3)
        C1("i") -= A("i");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        Cv[i] = alpha * Av[i] + beta * Cv[i];
    }

    return relative_difference(C1, C2);
}

double try_permute_rank2_ij()
{
    Dimension Cdims = {3, 4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j"}, {"i", "j"}, alpha, beta);
    else if (mode == 1)
        C1("ij") = A("ij");
    else if (mode == 2)
        C1("ij") += A("ij");
    else if (mode == 3)
        C1("ij") -= A("ij");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            Cv[i * Cdims[1] + j] =
                alpha * Av[i * Adims[1] + j] + beta * Cv[i * Cdims[1] + j];
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank2_ji()
{
    Dimension Cdims = {3, 4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 3};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j"}, {"j", "i"}, alpha, beta);
    else if (mode == 1)
        C1("ij") = A("ji");
    else if (mode == 2)
        C1("ij") += A("ji");
    else if (mode == 3)
        C1("ij") -= A("ji");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            Cv[i * Cdims[1] + j] =
                alpha * Av[j * Adims[1] + i] + beta * Cv[i * Cdims[1] + j];
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank3_ijk()
{
    Dimension Cdims = {3, 4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k"}, {"i", "j", "k"}, alpha, beta);
    else if (mode == 1)
        C1("ijk") = A("ijk");
    else if (mode == 2)
        C1("ijk") += A("ijk");
    else if (mode == 3)
        C1("ijk") -= A("ijk");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k] =
                    alpha * Av[i * Adims[1] * Adims[2] + j * Adims[2] + k] +
                    beta * Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k];
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank3_ikj()
{
    Dimension Cdims = {3, 4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 5, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k"}, {"i", "k", "j"}, alpha, beta);
    else if (mode == 1)
        C1("ijk") = A("ikj");
    else if (mode == 2)
        C1("ijk") += A("ikj");
    else if (mode == 3)
        C1("ijk") -= A("ikj");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k] =
                    alpha * Av[i * Adims[1] * Adims[2] + k * Adims[2] + j] +
                    beta * Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k];
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank3_jik()
{
    Dimension Cdims = {3, 4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 3, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k"}, {"j", "i", "k"}, alpha, beta);
    else if (mode == 1)
        C1("ijk") = A("jik");
    else if (mode == 2)
        C1("ijk") += A("jik");
    else if (mode == 3)
        C1("ijk") -= A("jik");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k] =
                    alpha * Av[j * Adims[1] * Adims[2] + i * Adims[2] + k] +
                    beta * Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k];
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank3_jki()
{
    Dimension Cdims = {3, 4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5, 3};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k"}, {"j", "k", "i"}, alpha, beta);
    else if (mode == 1)
        C1("ijk") = A("jki");
    else if (mode == 2)
        C1("ijk") += A("jki");
    else if (mode == 3)
        C1("ijk") -= A("jki");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k] =
                    alpha * Av[j * Adims[1] * Adims[2] + k * Adims[2] + i] +
                    beta * Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k];
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank3_kij()
{
    Dimension Cdims = {3, 4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {5, 3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k"}, {"k", "i", "j"}, alpha, beta);
    else if (mode == 1)
        C1("ijk") = A("kij");
    else if (mode == 2)
        C1("ijk") += A("kij");
    else if (mode == 3)
        C1("ijk") -= A("kij");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k] =
                    alpha * Av[k * Adims[1] * Adims[2] + i * Adims[2] + j] +
                    beta * Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k];
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank3_kji()
{
    Dimension Cdims = {3, 4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {5, 4, 3};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k"}, {"k", "j", "i"}, alpha, beta);
    else if (mode == 1)
        C1("ijk") = A("kji");
    else if (mode == 2)
        C1("ijk") += A("kji");
    else if (mode == 3)
        C1("ijk") -= A("kji");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k] =
                    alpha * Av[k * Adims[1] * Adims[2] + j * Adims[2] + i] +
                    beta * Cv[i * Cdims[1] * Cdims[2] + j * Cdims[2] + k];
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank4_ijkl()
{
    Dimension Cdims = {3, 4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4, 5, 6};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k", "l"}, {"i", "j", "k", "l"}, alpha, beta);
    else if (mode == 1)
        C1("ijkl") = A("ijkl");
    else if (mode == 2)
        C1("ijkl") += A("ijkl");
    else if (mode == 3)
        C1("ijkl") -= A("ijkl");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                for (size_t l = 0; l < Cdims[3]; l++)
                {
                    Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                       j * Cdims[2] * Cdims[3] + k * Cdims[3] + l] =
                        alpha * Av[i * Adims[1] * Adims[2] * Adims[3] +
                                   j * Adims[2] * Adims[3] + k * Adims[3] + l] +
                        beta * Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                                  j * Cdims[2] * Cdims[3] + k * Cdims[3] + l];
                }
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank4_ijlk()
{
    Dimension Cdims = {3, 4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 4, 6, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k", "l"}, {"i", "j", "l", "k"}, alpha, beta);
    else if (mode == 1)
        C1("ijkl") = A("ijlk");
    else if (mode == 2)
        C1("ijkl") += A("ijlk");
    else if (mode == 3)
        C1("ijkl") -= A("ijlk");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                for (size_t l = 0; l < Cdims[3]; l++)
                {
                    Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                       j * Cdims[2] * Cdims[3] + k * Cdims[3] + l] =
                        alpha * Av[i * Adims[1] * Adims[2] * Adims[3] +
                                   j * Adims[2] * Adims[3] + l * Adims[3] + k] +
                        beta * Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                                  j * Cdims[2] * Cdims[3] + k * Cdims[3] + l];
                }
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank4_jikl()
{
    Dimension Cdims = {3, 4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 3, 5, 6};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k", "l"}, {"j", "i", "k", "l"}, alpha, beta);
    else if (mode == 1)
        C1("ijkl") = A("jikl");
    else if (mode == 2)
        C1("ijkl") += A("jikl");
    else if (mode == 3)
        C1("ijkl") -= A("jikl");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                for (size_t l = 0; l < Cdims[3]; l++)
                {
                    Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                       j * Cdims[2] * Cdims[3] + k * Cdims[3] + l] =
                        alpha * Av[j * Adims[1] * Adims[2] * Adims[3] +
                                   i * Adims[2] * Adims[3] + k * Adims[3] + l] +
                        beta * Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                                  j * Cdims[2] * Cdims[3] + k * Cdims[3] + l];
                }
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank4_ikjl()
{
    Dimension Cdims = {3, 4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {3, 5, 4, 6};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k", "l"}, {"i", "k", "j", "l"}, alpha, beta);
    else if (mode == 1)
        C1("ijkl") = A("ikjl");
    else if (mode == 2)
        C1("ijkl") += A("ikjl");
    else if (mode == 3)
        C1("ijkl") -= A("ikjl");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                for (size_t l = 0; l < Cdims[3]; l++)
                {
                    Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                       j * Cdims[2] * Cdims[3] + k * Cdims[3] + l] =
                        alpha * Av[i * Adims[1] * Adims[2] * Adims[3] +
                                   k * Adims[2] * Adims[3] + j * Adims[3] + l] +
                        beta * Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                                  j * Cdims[2] * Cdims[3] + k * Cdims[3] + l];
                }
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_rank4_lkji()
{
    Dimension Cdims = {3, 4, 5, 6};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {6, 5, 4, 3};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    if (mode == 0)
        C1.permute(A, {"i", "j", "k", "l"}, {"l", "k", "j", "i"}, alpha, beta);
    else if (mode == 1)
        C1("ijkl") = A("lkji");
    else if (mode == 2)
        C1("ijkl") += A("lkji");
    else if (mode == 3)
        C1("ijkl") -= A("lkji");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Cv = C2.data();
    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            for (size_t k = 0; k < Cdims[2]; k++)
            {
                for (size_t l = 0; l < Cdims[3]; l++)
                {
                    Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                       j * Cdims[2] * Cdims[3] + k * Cdims[3] + l] =
                        alpha * Av[l * Adims[1] * Adims[2] * Adims[3] +
                                   k * Adims[2] * Adims[3] + j * Adims[3] + i] +
                        beta * Cv[i * Cdims[1] * Cdims[2] * Cdims[3] +
                                  j * Cdims[2] * Cdims[3] + k * Cdims[3] + l];
                }
            }
        }
    }

    return relative_difference(C1, C2);
}

double try_permute_label_fail()
{
    Dimension Cdims = {3, 4};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    C("ij") = A("i");

    return 0.0;
}

double try_permute_rank_fail()
{
    Dimension Cdims = {3, 4};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    C("ij") = A("i");

    return 0.0;
}

double try_permute_index_fail()
{
    Dimension Cdims = {3, 4};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    C("ij") = A("jk");

    return 0.0;
}

double try_permute_size_fail()
{
    Dimension Cdims = {3, 4};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    C("ij") = A("ji");

    return 0.0;
}

double try_contract_scalar()
{
    Dimension Cdims = {};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {}, {}, {}, alpha, beta);
    else if (mode == 1)
        C1("") = A("") * B("");
    else if (mode == 2)
        C1("") += A("") * B("");
    else if (mode == 3)
        C1("") -= A("") * B("");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    Cv[0] = alpha * Av[0] * Bv[0] + beta * Cv[0];

    return relative_difference(C1, C2);
}
double try_contract_hadamard()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {4};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"i"}, {"i"}, {"i"}, alpha, beta);
    else if (mode == 1)
        C1("i") = A("i") * B("i");
    else if (mode == 2)
        C1("i") += A("i") * B("i");
    else if (mode == 3)
        C1("i") -= A("i") * B("i");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cdims[0]; i++)
    {
        Cv[i] = alpha * Av[i] * Bv[i] + beta * Cv[i];
    }

    return relative_difference(C1, C2);
}
double try_contract_dot()
{
    Dimension Cdims = {};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {4};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {}, {"i"}, {"i"}, alpha, beta);
    else if (mode == 1)
        C1("") = A("i") * B("i");
    else if (mode == 2)
        C1("") += A("i") * B("i");
    else if (mode == 3)
        C1("") -= A("i") * B("i");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    Cv[0] = beta * Cv[0];
    for (size_t i = 0; i < Adims[0]; i++)
    {
        Cv[0] += alpha * Av[i] * Bv[i];
    }

    return relative_difference(C1, C2);
}
double try_contract_axpy1()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {4};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"i"}, {}, {"i"}, alpha, beta);
    else if (mode == 1)
        C1("i") = A("") * B("i");
    else if (mode == 2)
        C1("i") += A("") * B("i");
    else if (mode == 3)
        C1("i") -= A("") * B("i");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cdims[0]; i++)
    {
        Cv[i] = alpha * Av[0] * Bv[i] + beta * Cv[i];
    }

    return relative_difference(C1, C2);
}
double try_contract_axpy2()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"i"}, {"i"}, {}, alpha, beta);
    else if (mode == 1)
        C1("i") = A("i") * B("");
    else if (mode == 2)
        C1("i") += A("i") * B("");
    else if (mode == 3)
        C1("i") -= A("i") * B("");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cdims[0]; i++)
    {
        Cv[i] = alpha * Av[i] * Bv[0] + beta * Cv[i];
    }

    return relative_difference(C1, C2);
}
double try_contract_ger1()
{
    Dimension Cdims = {4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"i", "j"}, {"i"}, {"j"}, alpha, beta);
    else if (mode == 1)
        C1("ij") = A("i") * B("j");
    else if (mode == 2)
        C1("ij") += A("i") * B("j");
    else if (mode == 3)
        C1("ij") -= A("i") * B("j");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            Cv[i * Cdims[1] + j] =
                alpha * Av[i] * Bv[j] + beta * Cv[i * Cdims[1] + j];
        }
    }

    return relative_difference(C1, C2);
}
double try_contract_ger2()
{
    Dimension Cdims = {4, 5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {4};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"i", "j"}, {"j"}, {"i"}, alpha, beta);
    else if (mode == 1)
        C1("ij") = A("j") * B("i");
    else if (mode == 2)
        C1("ij") += A("j") * B("i");
    else if (mode == 3)
        C1("ij") -= A("j") * B("i");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Cdims[0]; i++)
    {
        for (size_t j = 0; j < Cdims[1]; j++)
        {
            Cv[i * Cdims[1] + j] =
                alpha * Av[j] * Bv[i] + beta * Cv[i * Cdims[1] + j];
        }
    }

    return relative_difference(C1, C2);
}
double try_contract_gemv1()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"i"}, {"i", "j"}, {"j"}, alpha, beta);
    else if (mode == 1)
        C1("i") = A("ij") * B("j");
    else if (mode == 2)
        C1("i") += A("ij") * B("j");
    else if (mode == 3)
        C1("i") -= A("ij") * B("j");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Adims[0]; i++)
    {
        Cv[i] = beta * Cv[i];
        for (size_t j = 0; j < Adims[1]; j++)
        {
            Cv[i] += alpha * Av[i * Adims[1] + j] * Bv[j];
        }
    }

    return relative_difference(C1, C2);
}
double try_contract_gemv2()
{
    Dimension Cdims = {4};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {5, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"i"}, {"j", "i"}, {"j"}, alpha, beta);
    else if (mode == 1)
        C1("i") = A("ji") * B("j");
    else if (mode == 2)
        C1("i") += A("ji") * B("j");
    else if (mode == 3)
        C1("i") -= A("ji") * B("j");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t i = 0; i < Adims[1]; i++)
    {
        Cv[i] = beta * Cv[i];
        for (size_t j = 0; j < Adims[0]; j++)
        {
            Cv[i] += alpha * Av[j * Adims[1] + i] * Bv[j];
        }
    }

    return relative_difference(C1, C2);
}
double try_contract_gemv3()
{
    Dimension Cdims = {5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {4, 5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"j"}, {"i"}, {"i", "j"}, alpha, beta);
    else if (mode == 1)
        C1("j") = A("i") * B("ij");
    else if (mode == 2)
        C1("j") += A("i") * B("ij");
    else if (mode == 3)
        C1("j") -= A("i") * B("ij");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t j = 0; j < Bdims[1]; j++)
    {
        Cv[j] = beta * Cv[j];
        for (size_t i = 0; i < Bdims[0]; i++)
        {
            Cv[j] += alpha * Av[i] * Bv[i * Bdims[1] + j];
        }
    }

    return relative_difference(C1, C2);
}
double try_contract_gemv4()
{
    Dimension Cdims = {5};
    Tensor C1 = Tensor::build(CoreTensor, "C1", Cdims);
    Tensor C2 = Tensor::build(CoreTensor, "C2", Cdims);
    initialize_random(C1, C2);

    Dimension Adims = {4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {5, 4};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    if (mode == 0)
        C1.contract(A, B, {"j"}, {"i"}, {"j", "i"}, alpha, beta);
    else if (mode == 1)
        C1("j") = A("i") * B("ji");
    else if (mode == 2)
        C1("j") += A("i") * B("ji");
    else if (mode == 3)
        C1("j") -= A("i") * B("ji");
    else
        throw std::runtime_error("Bad mode.");

    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();

    for (size_t j = 0; j < Bdims[0]; j++)
    {
        Cv[j] = beta * Cv[j];
        for (size_t i = 0; i < Bdims[1]; i++)
        {
            Cv[j] += alpha * Av[i] * Bv[j * Bdims[1] + i];
        }
    }

    return relative_difference(C1, C2);
}
double try_C_equal_A_B(std::string c_ind, std::string a_ind, std::string b_ind,
                       std::vector<int> c_dim, std::vector<int> a_dim,
                       std::vector<int> b_dim)
{
    std::vector<size_t> dims;
    dims.push_back(3);
    dims.push_back(4);
    dims.push_back(5);

    size_t ni = 3;
    size_t nj = 4;
    size_t nk = 5;

    Tensor A = Tensor::build(CoreTensor, "A", {dims[a_dim[0]], dims[a_dim[1]]});
    initialize_random(A);
    Tensor B = Tensor::build(CoreTensor, "B", {dims[b_dim[0]], dims[b_dim[1]]});
    initialize_random(B);
    Tensor C1 = Tensor::build(CoreTensor, "C1", {dims[c_dim[0]], dims[c_dim[1]]});
    Tensor C2 = Tensor::build(CoreTensor, "C2", {dims[c_dim[0]], dims[c_dim[1]]});
    initialize_random(C1, C2);

    std::vector<std::string> c_inds = {std::string(1, c_ind[0]),
                                       std::string(1, c_ind[1])};
    std::vector<std::string> a_inds = {std::string(1, a_ind[0]),
                                       std::string(1, a_ind[1])};
    std::vector<std::string> b_inds = {std::string(1, b_ind[0]),
                                       std::string(1, b_ind[1])};

    if (mode == 0)
        C1.contract(A, B, c_inds, a_inds, b_inds, alpha, beta);
    else if (mode == 1)
        C1(c_ind) = A(a_ind) * B(b_ind);
    else if (mode == 2)
        C1(c_ind) += A(a_ind) * B(b_ind);
    else if (mode == 3)
        C1(c_ind) -= A(a_ind) * B(b_ind);
    else
        throw std::runtime_error("Bad mode.");

    C2.scale(beta);
    std::vector<double> &Av = A.data();
    std::vector<double> &Bv = B.data();
    std::vector<double> &Cv = C2.data();
    std::vector<size_t> n(3);
    for (n[0] = 0; n[0] < ni; ++n[0])
    {
        for (n[1] = 0; n[1] < nj; ++n[1])
        {
            for (n[2] = 0; n[2] < nk; ++n[2])
            {
                size_t aind1 = n[a_dim[0]];
                size_t aind2 = n[a_dim[1]];
                size_t bind1 = n[b_dim[0]];
                size_t bind2 = n[b_dim[1]];
                size_t cind1 = n[c_dim[0]];
                size_t cind2 = n[c_dim[1]];
                Cv[cind1 * dims[c_dim[1]] + cind2] +=
                    alpha * Av[aind1 * dims[a_dim[1]] + aind2] *
                    Bv[bind1 * dims[b_dim[1]] + bind2];
            }
        }
    }

    return relative_difference(C1, C2);
}
double try_contract_gemm1()
{
    return try_C_equal_A_B("ij", "ik", "jk", {0, 1}, {0, 2}, {1, 2});
}
double try_contract_gemm2()
{
    return try_C_equal_A_B("ij", "ik", "kj", {0, 1}, {0, 2}, {2, 1});
}
double try_contract_gemm3()
{
    return try_C_equal_A_B("ij", "ki", "jk", {0, 1}, {2, 0}, {1, 2});
}
double try_contract_gemm4()
{
    return try_C_equal_A_B("ij", "ki", "kj", {0, 1}, {2, 0}, {2, 1});
}
double try_contract_gemm5()
{
    return try_C_equal_A_B("ji", "ik", "jk", {1, 0}, {0, 2}, {1, 2});
}
double try_contract_gemm6()
{
    return try_C_equal_A_B("ji", "ik", "kj", {1, 0}, {0, 2}, {2, 1});
}
double try_contract_gemm7()
{
    return try_C_equal_A_B("ji", "ki", "jk", {1, 0}, {2, 0}, {1, 2});
}
double try_contract_gemm8()
{
    return try_C_equal_A_B("ji", "ki", "kj", {1, 0}, {2, 0}, {2, 1});
}
double try_contract_label_fail()
{
    Dimension Cdims = {3, 4};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {3, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {4, 5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    C("i") = A("ik") * B("jk");

    return 0.0;
}
double try_contract_einsum_fail1()
{
    Dimension Cdims = {3};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {3, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {4, 5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    C("i") = A("ik") * B("jk");

    return 0.0;
}
double try_contract_einsum_fail2()
{
    Dimension Cdims = {3};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {3, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {4, 5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    C("ij") = A("ji") * B("jj");

    return 0.0;
}
double try_contract_size_fail1()
{
    Dimension Cdims = {2, 2, 4};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {2, 3, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {2, 4, 5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    C("Pij") = A("Pik") * B("Pjk");

    return 0.0;
}
double try_contract_size_fail2()
{
    Dimension Cdims = {2, 3, 3};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {2, 3, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {2, 4, 5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    C("Pij") = A("Pik") * B("Pjk");

    return 0.0;
}
double try_contract_size_fail3()
{
    Dimension Cdims = {2, 3, 4};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {2, 3, 4};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {2, 4, 5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    C("Pij") = A("Pik") * B("Pjk");

    return 0.0;
}
double try_contract_size_fail4()
{
    Dimension Cdims = {1, 3, 4};
    Tensor C = Tensor::build(CoreTensor, "C", Cdims);
    initialize_random(C);

    Dimension Adims = {2, 3, 5};
    Tensor A = Tensor::build(CoreTensor, "A", Adims);
    initialize_random(A);

    Dimension Bdims = {2, 4, 5};
    Tensor B = Tensor::build(CoreTensor, "B", Bdims);
    initialize_random(B);

    C("Pij") = A("Pik") * B("Pjk");

    return 0.0;
}

int main(int argc, char *argv[])
{
    printf(ANSI_COLOR_RESET);
    srand(time(NULL));
    ambit::initialize(argc, argv);

    bool success;

    printf("==> Simple Operations <==\n\n");
    printf("%s\n", std::string(82, '-').c_str());
    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed",
           "Delta");
    printf("%s\n", std::string(82, '-').c_str());
    success = true;
    success &=
        test_function(try_relative_difference, "Relative Difference", kExact);
    success &= test_function(try_1_norm, "1-Norm", kEpsilon);
    success &= test_function(try_2_norm, "2-Norm", kEpsilon);
    success &= test_function(try_inf_norm, "Inf-Norm", kEpsilon);
    success &= test_function(try_zero, "Zero", kExact);
    success &= test_function(try_copy, "Copy", kExact);
    success &= test_function(try_scale, "Scale", kExact);
    printf("%s\n", std::string(82, '-').c_str());
    printf("Tests: %s\n\n", success ? "All passed" : "Some failed");

    printf("==> Slice Operations <==\n\n");
    success = true;
    printf("%s\n", std::string(82, '-').c_str());
    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed",
           "Delta");
    mode = 0;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &=
        test_function(try_slice_rank0_same1, "Slice Rank-0 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same1, "Slice Rank-1 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same2, "Slice Rank-1 Same 2", kExact);
    success &=
        test_function(try_slice_rank1_diff1, "Slice Rank-1 Diff 1", kExact);
    success &=
        test_function(try_slice_rank1_diff2, "Slice Rank-1 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_same1, "Slice Rank-2 Same 1", kExact);
    success &=
        test_function(try_slice_rank2_same2, "Slice Rank-2 Same 2", kExact);
    success &=
        test_function(try_slice_rank2_same3, "Slice Rank-2 Same 3", kExact);
    success &=
        test_function(try_slice_rank2_diff1, "Slice Rank-2 Diff 1", kExact);
    success &=
        test_function(try_slice_rank2_diff2, "Slice Rank-2 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_diff3, "Slice Rank-2 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_same1, "Slice Rank-3 Same 1", kExact);
    success &=
        test_function(try_slice_rank3_same2, "Slice Rank-3 Same 2", kExact);
    success &=
        test_function(try_slice_rank3_same3, "Slice Rank-3 Same 3", kExact);
    success &=
        test_function(try_slice_rank3_same4, "Slice Rank-3 Same 4", kExact);
    success &=
        test_function(try_slice_rank3_diff1, "Slice Rank-3 Diff 1", kExact);
    success &=
        test_function(try_slice_rank3_diff2, "Slice Rank-3 Diff 2", kExact);
    success &=
        test_function(try_slice_rank3_diff3, "Slice Rank-3 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_diff4, "Slice Rank-3 Diff 4", kExact);
    mode = 0;
    alpha = random_double();
    beta = random_double();
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &=
        test_function(try_slice_rank0_same1, "Slice Rank-0 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same1, "Slice Rank-1 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same2, "Slice Rank-1 Same 2", kExact);
    success &=
        test_function(try_slice_rank1_diff1, "Slice Rank-1 Diff 1", kExact);
    success &=
        test_function(try_slice_rank1_diff2, "Slice Rank-1 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_same1, "Slice Rank-2 Same 1", kExact);
    success &=
        test_function(try_slice_rank2_same2, "Slice Rank-2 Same 2", kExact);
    success &=
        test_function(try_slice_rank2_same3, "Slice Rank-2 Same 3", kExact);
    success &=
        test_function(try_slice_rank2_diff1, "Slice Rank-2 Diff 1", kExact);
    success &=
        test_function(try_slice_rank2_diff2, "Slice Rank-2 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_diff3, "Slice Rank-2 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_same1, "Slice Rank-3 Same 1", kExact);
    success &=
        test_function(try_slice_rank3_same2, "Slice Rank-3 Same 2", kExact);
    success &=
        test_function(try_slice_rank3_same3, "Slice Rank-3 Same 3", kExact);
    success &=
        test_function(try_slice_rank3_same4, "Slice Rank-3 Same 4", kExact);
    success &=
        test_function(try_slice_rank3_diff1, "Slice Rank-3 Diff 1", kExact);
    success &=
        test_function(try_slice_rank3_diff2, "Slice Rank-3 Diff 2", kExact);
    success &=
        test_function(try_slice_rank3_diff3, "Slice Rank-3 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_diff4, "Slice Rank-3 Diff 4", kExact);
    mode = 1;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: =\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &=
        test_function(try_slice_rank0_same1, "Slice Rank-0 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same1, "Slice Rank-1 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same2, "Slice Rank-1 Same 2", kExact);
    success &=
        test_function(try_slice_rank1_diff1, "Slice Rank-1 Diff 1", kExact);
    success &=
        test_function(try_slice_rank1_diff2, "Slice Rank-1 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_same1, "Slice Rank-2 Same 1", kExact);
    success &=
        test_function(try_slice_rank2_same2, "Slice Rank-2 Same 2", kExact);
    success &=
        test_function(try_slice_rank2_same3, "Slice Rank-2 Same 3", kExact);
    success &=
        test_function(try_slice_rank2_diff1, "Slice Rank-2 Diff 1", kExact);
    success &=
        test_function(try_slice_rank2_diff2, "Slice Rank-2 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_diff3, "Slice Rank-2 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_same1, "Slice Rank-3 Same 1", kExact);
    success &=
        test_function(try_slice_rank3_same2, "Slice Rank-3 Same 2", kExact);
    success &=
        test_function(try_slice_rank3_same3, "Slice Rank-3 Same 3", kExact);
    success &=
        test_function(try_slice_rank3_same4, "Slice Rank-3 Same 4", kExact);
    success &=
        test_function(try_slice_rank3_diff1, "Slice Rank-3 Diff 1", kExact);
    success &=
        test_function(try_slice_rank3_diff2, "Slice Rank-3 Diff 2", kExact);
    success &=
        test_function(try_slice_rank3_diff3, "Slice Rank-3 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_diff4, "Slice Rank-3 Diff 4", kExact);
    mode = 2;
    alpha = 1.0;
    beta = 1.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: +=\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &=
        test_function(try_slice_rank0_same1, "Slice Rank-0 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same1, "Slice Rank-1 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same2, "Slice Rank-1 Same 2", kExact);
    success &=
        test_function(try_slice_rank1_diff1, "Slice Rank-1 Diff 1", kExact);
    success &=
        test_function(try_slice_rank1_diff2, "Slice Rank-1 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_same1, "Slice Rank-2 Same 1", kExact);
    success &=
        test_function(try_slice_rank2_same2, "Slice Rank-2 Same 2", kExact);
    success &=
        test_function(try_slice_rank2_same3, "Slice Rank-2 Same 3", kExact);
    success &=
        test_function(try_slice_rank2_diff1, "Slice Rank-2 Diff 1", kExact);
    success &=
        test_function(try_slice_rank2_diff2, "Slice Rank-2 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_diff3, "Slice Rank-2 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_same1, "Slice Rank-3 Same 1", kExact);
    success &=
        test_function(try_slice_rank3_same2, "Slice Rank-3 Same 2", kExact);
    success &=
        test_function(try_slice_rank3_same3, "Slice Rank-3 Same 3", kExact);
    success &=
        test_function(try_slice_rank3_same4, "Slice Rank-3 Same 4", kExact);
    success &=
        test_function(try_slice_rank3_diff1, "Slice Rank-3 Diff 1", kExact);
    success &=
        test_function(try_slice_rank3_diff2, "Slice Rank-3 Diff 2", kExact);
    success &=
        test_function(try_slice_rank3_diff3, "Slice Rank-3 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_diff4, "Slice Rank-3 Diff 4", kExact);
    mode = 3;
    alpha = -1.0;
    beta = 1.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: -=\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &=
        test_function(try_slice_rank0_same1, "Slice Rank-0 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same1, "Slice Rank-1 Same 1", kExact);
    success &=
        test_function(try_slice_rank1_same2, "Slice Rank-1 Same 2", kExact);
    success &=
        test_function(try_slice_rank1_diff1, "Slice Rank-1 Diff 1", kExact);
    success &=
        test_function(try_slice_rank1_diff2, "Slice Rank-1 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_same1, "Slice Rank-2 Same 1", kExact);
    success &=
        test_function(try_slice_rank2_same2, "Slice Rank-2 Same 2", kExact);
    success &=
        test_function(try_slice_rank2_same3, "Slice Rank-2 Same 3", kExact);
    success &=
        test_function(try_slice_rank2_diff1, "Slice Rank-2 Diff 1", kExact);
    success &=
        test_function(try_slice_rank2_diff2, "Slice Rank-2 Diff 2", kExact);
    success &=
        test_function(try_slice_rank2_diff3, "Slice Rank-2 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_same1, "Slice Rank-3 Same 1", kExact);
    success &=
        test_function(try_slice_rank3_same2, "Slice Rank-3 Same 2", kExact);
    success &=
        test_function(try_slice_rank3_same3, "Slice Rank-3 Same 3", kExact);
    success &=
        test_function(try_slice_rank3_same4, "Slice Rank-3 Same 4", kExact);
    success &=
        test_function(try_slice_rank3_diff1, "Slice Rank-3 Diff 1", kExact);
    success &=
        test_function(try_slice_rank3_diff2, "Slice Rank-3 Diff 2", kExact);
    success &=
        test_function(try_slice_rank3_diff3, "Slice Rank-3 Diff 3", kExact);
    success &=
        test_function(try_slice_rank3_diff4, "Slice Rank-3 Diff 4", kExact);
    printf("%s\n", std::string(82, '-').c_str());
    printf("Tests: %s\n\n", success ? "All Passed" : "Some Failed");

    printf("==> Slice Exceptions <==\n\n");
    success = true;
    printf("%s\n", std::string(82, '-').c_str());
    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed",
           "Delta");
    mode = 0;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &=
        test_function(try_slice_label_fail, "Slice Label Fail", kException);
    success &=
        test_function(try_slice_rank_fail, "Slice Rank Fail", kException);
    success &=
        test_function(try_slice_size_fail, "Slice Size Fail", kException);
    success &=
        test_function(try_slice_bounds_fail, "Slice Bounds Fail", kException);
    printf("%s\n", std::string(82, '-').c_str());
    printf("Tests: %s\n\n", success ? "All Passed" : "Some Failed");

    printf("==> Permute Operations <==\n\n");
    success = true;
    printf("%s\n", std::string(82, '-').c_str());
    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed",
           "Delta");
    mode = 0;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_permute_rank0, "Permute Rank-0", kExact);
    success &= test_function(try_permute_rank1_i, "Permute Rank-1 i", kExact);
    success &= test_function(try_permute_rank2_ij, "Permute Rank-2 ij", kExact);
    success &= test_function(try_permute_rank2_ji, "Permute Rank-2 ji", kExact);
    success &=
        test_function(try_permute_rank3_ijk, "Permute Rank-3 ijk", kExact);
    success &=
        test_function(try_permute_rank3_ikj, "Permute Rank-3 ikj", kExact);
    success &=
        test_function(try_permute_rank3_jik, "Permute Rank-3 jik", kExact);
    success &=
        test_function(try_permute_rank3_jki, "Permute Rank-3 jki", kExact);
    success &=
        test_function(try_permute_rank3_kij, "Permute Rank-3 kij", kExact);
    success &=
        test_function(try_permute_rank3_kji, "Permute Rank-3 kji", kExact);
    success &=
        test_function(try_permute_rank4_ijkl, "Permute Rank-4 ijkl", kExact);
    success &=
        test_function(try_permute_rank4_lkji, "Permute Rank-4 lkji", kExact);
    success &=
        test_function(try_permute_rank4_ijlk, "Permute Rank-4 ijlk", kExact);
    success &=
        test_function(try_permute_rank4_jikl, "Permute Rank-4 jikl", kExact);
    success &=
        test_function(try_permute_rank4_ikjl, "Permute Rank-4 ikjl", kExact);
    success &=
        test_function(try_permute_rank4_lkji, "Permute Rank-4 lkji", kExact);
    mode = 0;
    alpha = random_double();
    beta = random_double();
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_permute_rank0, "Permute Rank-0", kExact);
    success &= test_function(try_permute_rank1_i, "Permute Rank-1 i", kExact);
    success &= test_function(try_permute_rank2_ij, "Permute Rank-2 ij", kExact);
    success &= test_function(try_permute_rank2_ji, "Permute Rank-2 ji", kExact);
    success &=
        test_function(try_permute_rank3_ijk, "Permute Rank-3 ijk", kExact);
    success &=
        test_function(try_permute_rank3_ikj, "Permute Rank-3 ikj", kExact);
    success &=
        test_function(try_permute_rank3_jik, "Permute Rank-3 jik", kExact);
    success &=
        test_function(try_permute_rank3_jki, "Permute Rank-3 jki", kExact);
    success &=
        test_function(try_permute_rank3_kij, "Permute Rank-3 kij", kExact);
    success &=
        test_function(try_permute_rank3_kji, "Permute Rank-3 kji", kExact);
    success &=
        test_function(try_permute_rank4_ijkl, "Permute Rank-4 ijkl", kExact);
    success &=
        test_function(try_permute_rank4_ijlk, "Permute Rank-4 ijlk", kExact);
    success &=
        test_function(try_permute_rank4_jikl, "Permute Rank-4 jikl", kExact);
    success &=
        test_function(try_permute_rank4_ikjl, "Permute Rank-4 ikjl", kExact);
    success &=
        test_function(try_permute_rank4_lkji, "Permute Rank-4 lkji", kExact);
    mode = 1;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: =\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_permute_rank0, "Permute Rank-0", kExact);
    success &= test_function(try_permute_rank1_i, "Permute Rank-1 i", kExact);
    success &= test_function(try_permute_rank2_ij, "Permute Rank-2 ij", kExact);
    success &= test_function(try_permute_rank2_ji, "Permute Rank-2 ji", kExact);
    success &=
        test_function(try_permute_rank3_ijk, "Permute Rank-3 ijk", kExact);
    success &=
        test_function(try_permute_rank3_ikj, "Permute Rank-3 ikj", kExact);
    success &=
        test_function(try_permute_rank3_jik, "Permute Rank-3 jik", kExact);
    success &=
        test_function(try_permute_rank3_jki, "Permute Rank-3 jki", kExact);
    success &=
        test_function(try_permute_rank3_kij, "Permute Rank-3 kij", kExact);
    success &=
        test_function(try_permute_rank3_kji, "Permute Rank-3 kji", kExact);
    success &=
        test_function(try_permute_rank4_ijkl, "Permute Rank-4 ijkl", kExact);
    success &=
        test_function(try_permute_rank4_ijlk, "Permute Rank-4 ijlk", kExact);
    success &=
        test_function(try_permute_rank4_jikl, "Permute Rank-4 jikl", kExact);
    success &=
        test_function(try_permute_rank4_ikjl, "Permute Rank-4 ikjl", kExact);
    success &=
        test_function(try_permute_rank4_lkji, "Permute Rank-4 lkji", kExact);
    mode = 2;
    alpha = 1.0;
    beta = 1.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: +=\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_permute_rank0, "Permute Rank-0", kExact);
    success &= test_function(try_permute_rank1_i, "Permute Rank-1 i", kExact);
    success &= test_function(try_permute_rank2_ij, "Permute Rank-2 ij", kExact);
    success &= test_function(try_permute_rank2_ji, "Permute Rank-2 ji", kExact);
    success &=
        test_function(try_permute_rank3_ijk, "Permute Rank-3 ijk", kExact);
    success &=
        test_function(try_permute_rank3_ikj, "Permute Rank-3 ikj", kExact);
    success &=
        test_function(try_permute_rank3_jik, "Permute Rank-3 jik", kExact);
    success &=
        test_function(try_permute_rank3_jki, "Permute Rank-3 jki", kExact);
    success &=
        test_function(try_permute_rank3_kij, "Permute Rank-3 kij", kExact);
    success &=
        test_function(try_permute_rank3_kji, "Permute Rank-3 kji", kExact);
    success &=
        test_function(try_permute_rank4_ijkl, "Permute Rank-4 ijkl", kExact);
    success &=
        test_function(try_permute_rank4_ijlk, "Permute Rank-4 ijlk", kExact);
    success &=
        test_function(try_permute_rank4_jikl, "Permute Rank-4 jikl", kExact);
    success &=
        test_function(try_permute_rank4_ikjl, "Permute Rank-4 ikjl", kExact);
    success &=
        test_function(try_permute_rank4_lkji, "Permute Rank-4 lkji", kExact);
    mode = 3;
    alpha = -1.0;
    beta = 1.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: -=\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_permute_rank0, "Permute Rank-0", kExact);
    success &= test_function(try_permute_rank1_i, "Permute Rank-1 i", kExact);
    success &= test_function(try_permute_rank2_ij, "Permute Rank-2 ij", kExact);
    success &= test_function(try_permute_rank2_ji, "Permute Rank-2 ji", kExact);
    success &=
        test_function(try_permute_rank3_ijk, "Permute Rank-3 ijk", kExact);
    success &=
        test_function(try_permute_rank3_ikj, "Permute Rank-3 ikj", kExact);
    success &=
        test_function(try_permute_rank3_jik, "Permute Rank-3 jik", kExact);
    success &=
        test_function(try_permute_rank3_jki, "Permute Rank-3 jki", kExact);
    success &=
        test_function(try_permute_rank3_kij, "Permute Rank-3 kij", kExact);
    success &=
        test_function(try_permute_rank3_kji, "Permute Rank-3 kji", kExact);
    success &=
        test_function(try_permute_rank4_ijkl, "Permute Rank-4 ijkl", kExact);
    success &=
        test_function(try_permute_rank4_ijlk, "Permute Rank-4 ijlk", kExact);
    success &=
        test_function(try_permute_rank4_jikl, "Permute Rank-4 jikl", kExact);
    success &=
        test_function(try_permute_rank4_ikjl, "Permute Rank-4 ikjl", kExact);
    success &=
        test_function(try_permute_rank4_lkji, "Permute Rank-4 lkji", kExact);
    printf("%s\n", std::string(82, '-').c_str());
    printf("Tests: %s\n\n", success ? "All Passed" : "Some Failed");

    printf("==> Permute Exceptions <==\n\n");
    success = true;
    printf("%s\n", std::string(82, '-').c_str());
    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed",
           "Delta");
    mode = 0;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &=
        test_function(try_permute_label_fail, "Permute Label Fail", kException);
    success &=
        test_function(try_permute_rank_fail, "Permute Rank Fail", kException);
    success &=
        test_function(try_permute_index_fail, "Permute Index Fail", kException);
    success &=
        test_function(try_permute_size_fail, "Permute Size Fail", kException);
    printf("%s\n", std::string(82, '-').c_str());
    printf("Tests: %s\n\n", success ? "All Passed" : "Some Failed");

    printf("==> Contract Operations <==\n\n");
    success = true;
    printf("%s\n", std::string(82, '-').c_str());
    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed",
           "Delta");
    mode = 0;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_contract_scalar, "Contract scalar", kEpsilon);
    success &=
        test_function(try_contract_hadamard, "Contract hadamard", kEpsilon);
    success &= test_function(try_contract_dot, "Contract dot", kEpsilon);
    success &= test_function(try_contract_axpy1, "Contract axpy 1", kEpsilon);
    success &= test_function(try_contract_axpy2, "Contract axpy 2", kEpsilon);
    success &= test_function(try_contract_ger1, "Contract ger 1", kEpsilon);
    success &= test_function(try_contract_ger2, "Contract ger 2", kEpsilon);
    success &= test_function(try_contract_gemv1, "Contract gemv 1", kEpsilon);
    success &= test_function(try_contract_gemv2, "Contract gemv 2", kEpsilon);
    success &= test_function(try_contract_gemv3, "Contract gemv 3", kEpsilon);
    success &= test_function(try_contract_gemv4, "Contract gemv 4", kEpsilon);
    success &= test_function(try_contract_gemm1, "Contract gemm 1", kEpsilon);
    success &= test_function(try_contract_gemm2, "Contract gemm 2", kEpsilon);
    success &= test_function(try_contract_gemm3, "Contract gemm 3", kEpsilon);
    success &= test_function(try_contract_gemm4, "Contract gemm 4", kEpsilon);
    success &= test_function(try_contract_gemm5, "Contract gemm 5", kEpsilon);
    success &= test_function(try_contract_gemm6, "Contract gemm 6", kEpsilon);
    success &= test_function(try_contract_gemm7, "Contract gemm 7", kEpsilon);
    success &= test_function(try_contract_gemm8, "Contract gemm 8", kEpsilon);
    mode = 0;
    alpha = random_double();
    beta = random_double();
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_contract_scalar, "Contract scalar", kEpsilon);
    success &=
        test_function(try_contract_hadamard, "Contract hadamard", kEpsilon);
    success &= test_function(try_contract_dot, "Contract dot", kEpsilon);
    success &= test_function(try_contract_axpy1, "Contract axpy 1", kEpsilon);
    success &= test_function(try_contract_axpy2, "Contract axpy 2", kEpsilon);
    success &= test_function(try_contract_ger1, "Contract ger 1", kEpsilon);
    success &= test_function(try_contract_ger2, "Contract ger 2", kEpsilon);
    success &= test_function(try_contract_gemv1, "Contract gemv 1", kEpsilon);
    success &= test_function(try_contract_gemv2, "Contract gemv 2", kEpsilon);
    success &= test_function(try_contract_gemv3, "Contract gemv 3", kEpsilon);
    success &= test_function(try_contract_gemv4, "Contract gemv 4", kEpsilon);
    success &= test_function(try_contract_gemm1, "Contract gemm 1", kEpsilon);
    success &= test_function(try_contract_gemm2, "Contract gemm 2", kEpsilon);
    success &= test_function(try_contract_gemm3, "Contract gemm 3", kEpsilon);
    success &= test_function(try_contract_gemm4, "Contract gemm 4", kEpsilon);
    success &= test_function(try_contract_gemm5, "Contract gemm 5", kEpsilon);
    success &= test_function(try_contract_gemm6, "Contract gemm 6", kEpsilon);
    success &= test_function(try_contract_gemm7, "Contract gemm 7", kEpsilon);
    success &= test_function(try_contract_gemm8, "Contract gemm 8", kEpsilon);
    mode = 1;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: =\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_contract_scalar, "Contract scalar", kEpsilon);
    success &=
        test_function(try_contract_hadamard, "Contract hadamard", kEpsilon);
    success &= test_function(try_contract_dot, "Contract dot", kEpsilon);
    success &= test_function(try_contract_axpy1, "Contract axpy 1", kEpsilon);
    success &= test_function(try_contract_axpy2, "Contract axpy 2", kEpsilon);
    success &= test_function(try_contract_ger1, "Contract ger 1", kEpsilon);
    success &= test_function(try_contract_ger2, "Contract ger 2", kEpsilon);
    success &= test_function(try_contract_gemv1, "Contract gemv 1", kEpsilon);
    success &= test_function(try_contract_gemv2, "Contract gemv 2", kEpsilon);
    success &= test_function(try_contract_gemv3, "Contract gemv 3", kEpsilon);
    success &= test_function(try_contract_gemv4, "Contract gemv 4", kEpsilon);
    success &= test_function(try_contract_gemm1, "Contract gemm 1", kEpsilon);
    success &= test_function(try_contract_gemm2, "Contract gemm 2", kEpsilon);
    success &= test_function(try_contract_gemm3, "Contract gemm 3", kEpsilon);
    success &= test_function(try_contract_gemm4, "Contract gemm 4", kEpsilon);
    success &= test_function(try_contract_gemm5, "Contract gemm 5", kEpsilon);
    success &= test_function(try_contract_gemm6, "Contract gemm 6", kEpsilon);
    success &= test_function(try_contract_gemm7, "Contract gemm 7", kEpsilon);
    success &= test_function(try_contract_gemm8, "Contract gemm 8", kEpsilon);
    mode = 2;
    alpha = 1.0;
    beta = 1.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: +=\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_contract_scalar, "Contract scalar", kEpsilon);
    success &=
        test_function(try_contract_hadamard, "Contract hadamard", kEpsilon);
    success &= test_function(try_contract_dot, "Contract dot", kEpsilon);
    success &= test_function(try_contract_axpy1, "Contract axpy 1", kEpsilon);
    success &= test_function(try_contract_axpy2, "Contract axpy 2", kEpsilon);
    success &= test_function(try_contract_ger1, "Contract ger 1", kEpsilon);
    success &= test_function(try_contract_ger2, "Contract ger 2", kEpsilon);
    success &= test_function(try_contract_gemv1, "Contract gemv 1", kEpsilon);
    success &= test_function(try_contract_gemv2, "Contract gemv 2", kEpsilon);
    success &= test_function(try_contract_gemv3, "Contract gemv 3", kEpsilon);
    success &= test_function(try_contract_gemv4, "Contract gemv 4", kEpsilon);
    success &= test_function(try_contract_gemm1, "Contract gemm 1", kEpsilon);
    success &= test_function(try_contract_gemm2, "Contract gemm 2", kEpsilon);
    success &= test_function(try_contract_gemm3, "Contract gemm 3", kEpsilon);
    success &= test_function(try_contract_gemm4, "Contract gemm 4", kEpsilon);
    success &= test_function(try_contract_gemm5, "Contract gemm 5", kEpsilon);
    success &= test_function(try_contract_gemm6, "Contract gemm 6", kEpsilon);
    success &= test_function(try_contract_gemm7, "Contract gemm 7", kEpsilon);
    success &= test_function(try_contract_gemm8, "Contract gemm 8", kEpsilon);
    mode = 3;
    alpha = -1.0;
    beta = 1.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Operator Overloading: -=\n");
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_contract_scalar, "Contract scalar", kEpsilon);
    success &=
        test_function(try_contract_hadamard, "Contract hadamard", kEpsilon);
    success &= test_function(try_contract_dot, "Contract dot", kEpsilon);
    success &= test_function(try_contract_axpy1, "Contract axpy 1", kEpsilon);
    success &= test_function(try_contract_axpy2, "Contract axpy 2", kEpsilon);
    success &= test_function(try_contract_ger1, "Contract ger 1", kEpsilon);
    success &= test_function(try_contract_ger2, "Contract ger 2", kEpsilon);
    success &= test_function(try_contract_gemv1, "Contract gemv 1", kEpsilon);
    success &= test_function(try_contract_gemv2, "Contract gemv 2", kEpsilon);
    success &= test_function(try_contract_gemv3, "Contract gemv 3", kEpsilon);
    success &= test_function(try_contract_gemv4, "Contract gemv 4", kEpsilon);
    success &= test_function(try_contract_gemm1, "Contract gemm 1", kEpsilon);
    success &= test_function(try_contract_gemm2, "Contract gemm 2", kEpsilon);
    success &= test_function(try_contract_gemm3, "Contract gemm 3", kEpsilon);
    success &= test_function(try_contract_gemm4, "Contract gemm 4", kEpsilon);
    success &= test_function(try_contract_gemm5, "Contract gemm 5", kEpsilon);
    success &= test_function(try_contract_gemm6, "Contract gemm 6", kEpsilon);
    success &= test_function(try_contract_gemm7, "Contract gemm 7", kEpsilon);
    success &= test_function(try_contract_gemm8, "Contract gemm 8", kEpsilon);
    printf("%s\n", std::string(82, '-').c_str());
    printf("Tests: %s\n\n", success ? "All Passed" : "Some Failed");

    printf("==> Contract Exceptions <==\n\n");
    success = true;
    printf("%s\n", std::string(82, '-').c_str());
    printf("%-50s %-9s %-9s %11s\n", "Description", "Expected", "Observed",
           "Delta");
    mode = 0;
    alpha = 1.0;
    beta = 0.0;
    printf("%s\n", std::string(82, '-').c_str());
    printf("Explicit: alpha = %11.3E, beta = %11.3E\n", alpha, beta);
    printf("%s\n", std::string(82, '-').c_str());
    success &= test_function(try_contract_label_fail, "Contract Label Fail",
                             kException);
    success &= test_function(try_contract_einsum_fail1,
                             "Contract Einsum Fail 1", kException);
    success &= test_function(try_contract_einsum_fail2,
                             "Contract Einsum Fail 2", kException);
    success &= test_function(try_contract_size_fail1, "Contract Size Fail 1",
                             kException);
    success &= test_function(try_contract_size_fail2, "Contract Size Fail 2",
                             kException);
    success &= test_function(try_contract_size_fail3, "Contract Size Fail 3",
                             kException);
    success &= test_function(try_contract_size_fail4, "Contract Size Fail 4",
                             kException);
    printf("%s\n", std::string(82, '-').c_str());
    printf("Tests: %s\n\n", success ? "All Passed" : "Some Failed");

    ambit::finalize();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
