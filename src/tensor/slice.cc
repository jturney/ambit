#include "slice.h"
#include "math/math.h"
#include <ambit/timer.h>
#include <string.h>

namespace ambit
{

void slice(TensorImplPtr C, ConstTensorImplPtr A, const IndexRange &Cinds,
           const IndexRange &Ainds, double alpha, double beta)
{
    // => Error Checking <= //

    for (size_t ind = 0L; ind < C->rank(); ind++)
    {
        size_t Asize = Ainds[ind][1] - Ainds[ind][0];
        size_t Csize = Cinds[ind][1] - Cinds[ind][0];
        if (Asize != Csize)
        {
            throw std::runtime_error(
                "Slice range sizes must agree between tensors A and C.");
        }
    }

    // => Type Logic <= //

    if (C->type() == CoreTensor and A->type() == CoreTensor)
    {
        slice(dynamic_cast<CoreTensorImplPtr>(C),
              dynamic_cast<ConstCoreTensorImplPtr>(A), Cinds, Ainds, alpha,
              beta);
    }
    else if (C->type() == CoreTensor and A->type() == DiskTensor)
    {
        slice(dynamic_cast<CoreTensorImplPtr>(C),
              dynamic_cast<ConstDiskTensorImplPtr>(A), Cinds, Ainds, alpha,
              beta);
    }
    else if (C->type() == DiskTensor and A->type() == CoreTensor)
    {
        slice(dynamic_cast<DiskTensorImplPtr>(C),
              dynamic_cast<ConstCoreTensorImplPtr>(A), Cinds, Ainds, alpha,
              beta);
    }
    else if (C->type() == DiskTensor and A->type() == DiskTensor)
    {
        slice(dynamic_cast<DiskTensorImplPtr>(C),
              dynamic_cast<ConstDiskTensorImplPtr>(A), Cinds, Ainds, alpha,
              beta);

#ifdef HAVE_CYCLOPS
    }
    else if (C->type() == CoreTensor and A->type() == DistributedTensor)
    {
        slice((CoreTensorImplPtr)C, (ConstCyclopsTensorImplPtr)A, Cinds, Ainds,
              alpha, beta);
    }
    else if (C->type() == DistributedTensor and A->type() == CoreTensor)
    {
        slice((CyclopsTensorImplPtr)C, (ConstCoreTensorImplPtr)A, Cinds, Ainds,
              alpha, beta);
    }
    else if (C->type() == DistributedTensor and A->type() == DistributedTensor)
    {
        slice((CyclopsTensorImplPtr)C, (ConstCyclopsTensorImplPtr)A, Cinds,
              Ainds, alpha, beta);
#endif
    }
    else
    {
        throw std::runtime_error("Slice cannot handle this type pairing.");
    }
}

void slice(CoreTensorImplPtr C, ConstCoreTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha,
           double beta)
{
    timer::timer_push("slice Core -> Core");
    /// Data pointers
    double *Cp = C->data().data();
    double *Ap = const_cast<CoreTensorImplPtr>(A)->data().data();

    // => Special Case: Rank-0 <= //

    if (C->rank() == 0)
    {
        Cp[0] = alpha * Ap[0] + beta * Cp[0];
    }
    else
    {
        // => Index Logic <= //

        /// Sizes of stripes
        std::vector<size_t> sizes(C->rank(), 0L);
        for (size_t ind = 0L; ind < C->rank(); ind++)
        {
            sizes[ind] = Cinds[ind][1] - Cinds[ind][0];
        }

        /// Size of contiguous DAXPY call
        int fast_dims = 1;
        size_t fast_size = sizes[C->rank() - 1];
        for (int ind = ((int)C->rank()) - 2; ind >= 0; ind--)
        {
            if (sizes[ind + 1] == A->dims()[ind + 1] &&
                sizes[ind + 1] == C->dims()[ind + 1])
            {
                fast_dims++;
                fast_size *= sizes[ind];
            }
            else
            {
                break;
            }
        }

        int slow_dims = C->rank() - fast_dims;
        size_t slow_size = 1L;
        for (int dim = 0; dim < slow_dims; dim++)
        {
            slow_size *= sizes[dim];
        }

        std::vector<size_t> Astrides(C->rank());
        Astrides[C->rank() - 1] = 1L;
        for (int ind = ((int)C->rank() - 2); ind >= 0; ind--)
        {
            Astrides[ind] = Astrides[ind + 1] * A->dims()[ind + 1];
        }

        std::vector<size_t> Cstrides(C->rank());
        Cstrides[C->rank() - 1] = 1L;
        for (int ind = ((int)C->rank() - 2); ind >= 0; ind--)
        {
            Cstrides[ind] = Cstrides[ind + 1] * C->dims()[ind + 1];
        }

        // => Slice Operation <= //

        if (slow_dims == 0)
        {
            double *Atp = Ap + Ainds[slow_dims][0] * Astrides[slow_dims];
            double *Ctp = Cp + Cinds[slow_dims][0] * Cstrides[slow_dims];
            C_DSCAL(fast_size, beta, Ctp, 1);
            C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
        }
        else if (slow_dims == 1)
        {
#pragma omp parallel for
            for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++)
            {
                double *Atp = Ap + (Ainds[0][0] + ind0) * Astrides[0] +
                              Ainds[slow_dims][0] * Astrides[slow_dims];
                double *Ctp = Cp + (Cinds[0][0] + ind0) * Cstrides[0] +
                              Cinds[slow_dims][0] * Cstrides[slow_dims];
                C_DSCAL(fast_size, beta, Ctp, 1);
                C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
            }
        }
        else if (slow_dims == 2)
        {
#pragma omp parallel for
            for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++)
            {
                for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++)
                {
                    double *Atp = Ap + (Ainds[0][0] + ind0) * Astrides[0] +
                                  (Ainds[1][0] + ind1) * Astrides[1] +
                                  Ainds[slow_dims][0] * Astrides[slow_dims];
                    double *Ctp = Cp + (Cinds[0][0] + ind0) * Cstrides[0] +
                                  (Cinds[1][0] + ind1) * Cstrides[1] +
                                  Cinds[slow_dims][0] * Cstrides[slow_dims];
                    C_DSCAL(fast_size, beta, Ctp, 1);
                    C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                }
            }
        }
        else if (slow_dims == 3)
        {
#pragma omp parallel for
            for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++)
            {
                for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++)
                {
                    for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++)
                    {
                        double *Atp = Ap + (Ainds[0][0] + ind0) * Astrides[0] +
                                      (Ainds[1][0] + ind1) * Astrides[1] +
                                      (Ainds[2][0] + ind2) * Astrides[2] +
                                      Ainds[slow_dims][0] * Astrides[slow_dims];
                        double *Ctp = Cp + (Cinds[0][0] + ind0) * Cstrides[0] +
                                      (Cinds[1][0] + ind1) * Cstrides[1] +
                                      (Cinds[2][0] + ind2) * Cstrides[2] +
                                      Cinds[slow_dims][0] * Cstrides[slow_dims];
                        C_DSCAL(fast_size, beta, Ctp, 1);
                        C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                    }
                }
            }
        }
        else if (slow_dims == 4)
        {
#pragma omp parallel for
            for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++)
            {
                for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++)
                {
                    for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++)
                    {
                        for (size_t ind3 = 0L; ind3 < sizes[3]; ind3++)
                        {
                            double *Atp =
                                Ap + (Ainds[0][0] + ind0) * Astrides[0] +
                                (Ainds[1][0] + ind1) * Astrides[1] +
                                (Ainds[2][0] + ind2) * Astrides[2] +
                                (Ainds[3][0] + ind3) * Astrides[3] +
                                Ainds[slow_dims][0] * Astrides[slow_dims];
                            double *Ctp =
                                Cp + (Cinds[0][0] + ind0) * Cstrides[0] +
                                (Cinds[1][0] + ind1) * Cstrides[1] +
                                (Cinds[2][0] + ind2) * Cstrides[2] +
                                (Cinds[3][0] + ind3) * Cstrides[3] +
                                Cinds[slow_dims][0] * Cstrides[slow_dims];
                            C_DSCAL(fast_size, beta, Ctp, 1);
                            C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                        }
                    }
                }
            }
        }
        else if (slow_dims == 5)
        {
#pragma omp parallel for
            for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++)
            {
                for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++)
                {
                    for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++)
                    {
                        for (size_t ind3 = 0L; ind3 < sizes[3]; ind3++)
                        {
                            for (size_t ind4 = 0L; ind4 < sizes[4]; ind4++)
                            {
                                double *Atp =
                                    Ap + (Ainds[0][0] + ind0) * Astrides[0] +
                                    (Ainds[1][0] + ind1) * Astrides[1] +
                                    (Ainds[2][0] + ind2) * Astrides[2] +
                                    (Ainds[3][0] + ind3) * Astrides[3] +
                                    (Ainds[4][0] + ind4) * Astrides[4] +
                                    Ainds[slow_dims][0] * Astrides[slow_dims];
                                double *Ctp =
                                    Cp + (Cinds[0][0] + ind0) * Cstrides[0] +
                                    (Cinds[1][0] + ind1) * Cstrides[1] +
                                    (Cinds[2][0] + ind2) * Cstrides[2] +
                                    (Cinds[3][0] + ind3) * Cstrides[3] +
                                    (Cinds[4][0] + ind4) * Cstrides[4] +
                                    Cinds[slow_dims][0] * Cstrides[slow_dims];
                                C_DSCAL(fast_size, beta, Ctp, 1);
                                C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                            }
                        }
                    }
                }
            }
        }
        else if (slow_dims == 6)
        {
#pragma omp parallel for
            for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++)
            {
                for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++)
                {
                    for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++)
                    {
                        for (size_t ind3 = 0L; ind3 < sizes[3]; ind3++)
                        {
                            for (size_t ind4 = 0L; ind4 < sizes[4]; ind4++)
                            {
                                for (size_t ind5 = 0L; ind5 < sizes[5]; ind5++)
                                {
                                    double *Atp =
                                        Ap +
                                        (Ainds[0][0] + ind0) * Astrides[0] +
                                        (Ainds[1][0] + ind1) * Astrides[1] +
                                        (Ainds[2][0] + ind2) * Astrides[2] +
                                        (Ainds[3][0] + ind3) * Astrides[3] +
                                        (Ainds[4][0] + ind4) * Astrides[4] +
                                        (Ainds[5][0] + ind5) * Astrides[5] +
                                        Ainds[slow_dims][0] *
                                            Astrides[slow_dims];
                                    double *Ctp =
                                        Cp +
                                        (Cinds[0][0] + ind0) * Cstrides[0] +
                                        (Cinds[1][0] + ind1) * Cstrides[1] +
                                        (Cinds[2][0] + ind2) * Cstrides[2] +
                                        (Cinds[3][0] + ind3) * Cstrides[3] +
                                        (Cinds[4][0] + ind4) * Cstrides[4] +
                                        (Cinds[5][0] + ind5) * Cstrides[5] +
                                        Cinds[slow_dims][0] *
                                            Cstrides[slow_dims];
                                    C_DSCAL(fast_size, beta, Ctp, 1);
                                    C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (slow_dims == 7)
        {
#pragma omp parallel for
            for (size_t ind0 = 0L; ind0 < sizes[0]; ind0++)
            {
                for (size_t ind1 = 0L; ind1 < sizes[1]; ind1++)
                {
                    for (size_t ind2 = 0L; ind2 < sizes[2]; ind2++)
                    {
                        for (size_t ind3 = 0L; ind3 < sizes[3]; ind3++)
                        {
                            for (size_t ind4 = 0L; ind4 < sizes[4]; ind4++)
                            {
                                for (size_t ind5 = 0L; ind5 < sizes[5]; ind5++)
                                {
                                    for (size_t ind6 = 0L; ind6 < sizes[6];
                                         ind6++)
                                    {
                                        double *Atp =
                                            Ap +
                                            (Ainds[0][0] + ind0) * Astrides[0] +
                                            (Ainds[1][0] + ind1) * Astrides[1] +
                                            (Ainds[2][0] + ind2) * Astrides[2] +
                                            (Ainds[3][0] + ind3) * Astrides[3] +
                                            (Ainds[4][0] + ind4) * Astrides[4] +
                                            (Ainds[5][0] + ind5) * Astrides[5] +
                                            (Ainds[6][0] + ind6) * Astrides[6] +
                                            Ainds[slow_dims][0] *
                                                Astrides[slow_dims];
                                        double *Ctp =
                                            Cp +
                                            (Cinds[0][0] + ind0) * Cstrides[0] +
                                            (Cinds[1][0] + ind1) * Cstrides[1] +
                                            (Cinds[2][0] + ind2) * Cstrides[2] +
                                            (Cinds[3][0] + ind3) * Cstrides[3] +
                                            (Cinds[4][0] + ind4) * Cstrides[4] +
                                            (Cinds[5][0] + ind5) * Cstrides[5] +
                                            (Cinds[6][0] + ind6) * Cstrides[6] +
                                            Cinds[slow_dims][0] *
                                                Cstrides[slow_dims];
                                        C_DSCAL(fast_size, beta, Ctp, 1);
                                        C_DAXPY(fast_size, alpha, Atp, 1, Ctp,
                                                1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (size_t ind = 0L; ind < slow_size; ind++)
            {
                double *Ctp = Cp;
                double *Atp = Ap;
                size_t num = ind;
                for (int dim = slow_dims - 1; dim >= 0; dim--)
                {
                    size_t val = num % sizes[dim]; // value of the dim-th index
                    num /= sizes[dim];
                    Atp += (Ainds[dim][0] + val) * Astrides[dim];
                    Ctp += (Cinds[dim][0] + val) * Cstrides[dim];
                }
                Atp += Ainds[slow_dims][0] * Astrides[slow_dims];
                Ctp += Cinds[slow_dims][0] * Cstrides[slow_dims];
                C_DSCAL(fast_size, beta, Ctp, 1);
                C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
            }
        }
    }

    timer::timer_pop();
}
void slice(CoreTensorImplPtr C, ConstDiskTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha,
           double beta)
{
    timer::timer_push("slice Disk -> Core");

    /// Data pointers
    double *Cp = C->data().data();
    FILE *Af = ((DiskTensorImplPtr)A)->fh();

    // => Special Case: Rank-0 <= //

    if (C->rank() == 0)
    {
        double Ap = 0.0;
        fseek(Af, 0L, SEEK_SET);
        fread(&Ap, sizeof(double), 1L, Af);
        fseek(Af, 0L, SEEK_SET);
        Cp[0] = alpha * Ap + beta * Cp[0];
    }
    else
    {
        // => Index Logic <= //

        /// Sizes of stripes
        vector<size_t> sizes(C->rank(), 0L);
        for (size_t ind = 0L; ind < C->rank(); ind++)
        {
            sizes[ind] = Cinds[ind][1] - Cinds[ind][0];
        }

        /// Size of contiguous DAXPY call
        int fast_dims = 1;
        size_t fast_size = sizes[C->rank() - 1];
        for (int ind = ((int)C->rank()) - 2; ind >= 0; ind--)
        {
            if (sizes[ind + 1] == A->dims()[ind + 1] &&
                sizes[ind + 1] == C->dims()[ind + 1])
            {
                if (fast_size * sizes[ind] <= disk_buffer__)
                {
                    fast_dims++;
                    fast_size *= sizes[ind];
                }
                else
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }

        int slow_dims = C->rank() - fast_dims;
        size_t slow_size = 1L;
        for (int dim = 0; dim < slow_dims; dim++)
        {
            slow_size *= sizes[dim];
        }

        vector<size_t> Astrides(C->rank());
        Astrides[C->rank() - 1] = 1L;
        for (int ind = ((int)C->rank() - 2); ind >= 0; ind--)
        {
            Astrides[ind] = Astrides[ind + 1] * A->dims()[ind + 1];
        }

        vector<size_t> Cstrides(C->rank());
        Cstrides[C->rank() - 1] = 1L;
        for (int ind = ((int)C->rank() - 2); ind >= 0; ind--)
        {
            Cstrides[ind] = Cstrides[ind + 1] * C->dims()[ind + 1];
        }

        /// Buffer of A
        double *Ap = new double[fast_size];
        memset(Ap, '\0', sizeof(double) * fast_size);

        // => Slice Operation <= //

        for (size_t ind = 0L; ind < slow_size; ind++)
        {
            size_t num = ind;
            size_t Aoff = 0L;
            size_t Coff = 0L;
            for (int dim = slow_dims - 1; dim >= 0; dim--)
            {
                size_t val = num % sizes[dim]; // value of the dim-th index
                num /= sizes[dim];
                Aoff += (Ainds[dim][0] + val) * Astrides[dim];
                Coff += (Cinds[dim][0] + val) * Cstrides[dim];
            }
            Aoff += Ainds[slow_dims][0] * Astrides[slow_dims];
            Coff += Cinds[slow_dims][0] * Cstrides[slow_dims];

            double *Ctp = Cp + Coff;
            double *Atp = Ap;
            fseek(Af, sizeof(double) * Aoff, SEEK_SET);
            fread(Atp, sizeof(double), fast_size, Af);

            C_DSCAL(fast_size, beta, Ctp, 1);
            C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
        }

        delete[] Ap;
        fseek(Af, 0L, SEEK_SET);
    }

    timer::timer_pop();
}
void slice(DiskTensorImplPtr C, ConstCoreTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha,
           double beta)
{
    timer::timer_push("slice Core -> Disk");

    /// Data pointers
    FILE *Cf = C->fh();
    double *Ap = ((CoreTensorImplPtr)A)->data().data();

    // => Special Case: Rank-0 <= //

    if (C->rank() == 0)
    {
        double Cp = 0.0;
        if (beta != 0.0)
        {
            fseek(Cf, 0L, SEEK_SET);
            fread(&Cp, sizeof(double), 1L, Cf);
        }
        Cp = alpha * Ap[0] + beta * Cp;
        fseek(Cf, 0L, SEEK_SET);
        fwrite(&Cp, sizeof(double), 1L, Cf);
        fseek(Cf, 0L, SEEK_SET);
    }
    else
    {
        // => Index Logic <= //

        /// Sizes of stripes
        vector<size_t> sizes(C->rank(), 0L);
        for (size_t ind = 0L; ind < C->rank(); ind++)
        {
            sizes[ind] = Cinds[ind][1] - Cinds[ind][0];
        }

        /// Size of contiguous DAXPY call
        int fast_dims = 1;
        size_t fast_size = sizes[C->rank() - 1];
        for (int ind = ((int)C->rank()) - 2; ind >= 0; ind--)
        {
            if (sizes[ind + 1] == A->dims()[ind + 1] &&
                sizes[ind + 1] == C->dims()[ind + 1])
            {
                if (fast_size * sizes[ind] <= disk_buffer__)
                {
                    fast_dims++;
                    fast_size *= sizes[ind];
                }
                else
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }

        int slow_dims = C->rank() - fast_dims;
        size_t slow_size = 1L;
        for (int dim = 0; dim < slow_dims; dim++)
        {
            slow_size *= sizes[dim];
        }

        vector<size_t> Astrides(C->rank());
        Astrides[C->rank() - 1] = 1L;
        for (int ind = ((int)C->rank() - 2); ind >= 0; ind--)
        {
            Astrides[ind] = Astrides[ind + 1] * A->dims()[ind + 1];
        }

        vector<size_t> Cstrides(C->rank());
        Cstrides[C->rank() - 1] = 1L;
        for (int ind = ((int)C->rank() - 2); ind >= 0; ind--)
        {
            Cstrides[ind] = Cstrides[ind + 1] * C->dims()[ind + 1];
        }

        /// Buffer of C
        double *Cp = new double[fast_size];
        memset(Cp, '\0', sizeof(double) * fast_size);

        // => Slice Operation <= //

        for (size_t ind = 0L; ind < slow_size; ind++)
        {
            size_t num = ind;
            size_t Aoff = 0L;
            size_t Coff = 0L;
            for (int dim = slow_dims - 1; dim >= 0; dim--)
            {
                size_t val = num % sizes[dim]; // value of the dim-th index
                num /= sizes[dim];
                Aoff += (Ainds[dim][0] + val) * Astrides[dim];
                Coff += (Cinds[dim][0] + val) * Cstrides[dim];
            }
            Aoff += Ainds[slow_dims][0] * Astrides[slow_dims];
            Coff += Cinds[slow_dims][0] * Cstrides[slow_dims];

            double *Ctp = Cp;
            double *Atp = Ap + Aoff;

            if (beta != 0.0)
            {
                fseek(Cf, sizeof(double) * Coff, SEEK_SET);
                fread(Ctp, sizeof(double), fast_size, Cf);
            }

            C_DSCAL(fast_size, beta, Ctp, 1);
            C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);

            fseek(Cf, sizeof(double) * Coff, SEEK_SET);
            fwrite(Ctp, sizeof(double), fast_size, Cf);
        }

        delete[] Cp;
        fseek(Cf, 0L, SEEK_SET);
    }

    timer::timer_pop();
}
void slice(DiskTensorImplPtr C, ConstDiskTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha,
           double beta)
{
    timer::timer_push("slice Disk -> Disk");

    /// Data pointers
    FILE *Cf = C->fh();
    FILE *Af = A->fh();

    // => Special Case: Rank-0 <= //

    if (C->rank() == 0)
    {
        double Cp = 0.0;
        double Ap;
        if (beta != 0.0)
        {
            fseek(Cf, 0L, SEEK_SET);
            fread(&Cp, sizeof(double), 1L, Cf);
        }
        fseek(Af, 0L, SEEK_SET);
        fread(&Ap, sizeof(double), 1L, Af);
        Cp = alpha * Ap + beta * Cp;
        fseek(Cf, 0L, SEEK_SET);
        fwrite(&Cp, sizeof(double), 1L, Cf);
        fseek(Cf, 0L, SEEK_SET);
        fseek(Af, 0L, SEEK_SET);
    }
    else
    {
        // => Index Logic <= //

        /// Sizes of stripes
        vector<size_t> sizes(C->rank(), 0L);
        for (size_t ind = 0L; ind < C->rank(); ind++)
        {
            sizes[ind] = Cinds[ind][1] - Cinds[ind][0];
        }

        /// Size of contiguous DAXPY call
        int fast_dims = 1;
        size_t fast_size = sizes[C->rank() - 1];
        for (int ind = ((int)C->rank()) - 2; ind >= 0; ind--)
        {
            if (sizes[ind + 1] == A->dims()[ind + 1] &&
                sizes[ind + 1] == C->dims()[ind + 1])
            {
                if (fast_size * sizes[ind] <= disk_buffer__)
                {
                    fast_dims++;
                    fast_size *= sizes[ind];
                }
                else
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }

        int slow_dims = C->rank() - fast_dims;
        size_t slow_size = 1L;
        for (int dim = 0; dim < slow_dims; dim++)
        {
            slow_size *= sizes[dim];
        }

        vector<size_t> Astrides(C->rank());
        Astrides[C->rank() - 1] = 1L;
        for (int ind = ((int)C->rank() - 2); ind >= 0; ind--)
        {
            Astrides[ind] = Astrides[ind + 1] * A->dims()[ind + 1];
        }

        vector<size_t> Cstrides(C->rank());
        Cstrides[C->rank() - 1] = 1L;
        for (int ind = ((int)C->rank() - 2); ind >= 0; ind--)
        {
            Cstrides[ind] = Cstrides[ind + 1] * C->dims()[ind + 1];
        }

        /// Buffer of A
        double *Ap = new double[fast_size];
        memset(Ap, '\0', sizeof(double) * fast_size);
        /// Buffer of C
        double *Cp = new double[fast_size];
        memset(Cp, '\0', sizeof(double) * fast_size);

        // => Slice Operation <= //

        for (size_t ind = 0L; ind < slow_size; ind++)
        {
            size_t num = ind;
            size_t Aoff = 0L;
            size_t Coff = 0L;
            for (int dim = slow_dims - 1; dim >= 0; dim--)
            {
                size_t val = num % sizes[dim]; // value of the dim-th index
                num /= sizes[dim];
                Aoff += (Ainds[dim][0] + val) * Astrides[dim];
                Coff += (Cinds[dim][0] + val) * Cstrides[dim];
            }
            Aoff += Ainds[slow_dims][0] * Astrides[slow_dims];
            Coff += Cinds[slow_dims][0] * Cstrides[slow_dims];

            double *Ctp = Cp;
            double *Atp = Ap;

            if (beta != 0.0)
            {
                fseek(Cf, sizeof(double) * Coff, SEEK_SET);
                fread(Ctp, sizeof(double), fast_size, Cf);
            }

            fseek(Af, sizeof(double) * Aoff, SEEK_SET);
            fread(Atp, sizeof(double), fast_size, Af);

            C_DSCAL(fast_size, beta, Ctp, 1);
            C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);

            fseek(Cf, sizeof(double) * Coff, SEEK_SET);
            fwrite(Ctp, sizeof(double), fast_size, Cf);
        }

        delete[] Ap;
        delete[] Cp;
        fseek(Af, 0L, SEEK_SET);
        fseek(Cf, 0L, SEEK_SET);
    }

    timer::timer_pop();
}

#ifdef HAVE_CYCLOPS
void slice(CoreTensorImplPtr C, ConstCyclopsTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha,
           double beta)
{
    timer::timer_push("slice Cyclops -> Core");

    if (C->rank() == 0)
    {
        double Av = ((CTF_Scalar *)A->cyclops())->get_val();
        double Cv = C->data()[0];
        C->data()[0] = alpha * Av + beta * Cv;
    }
    else
    {

        long_int numel = 1L;
        Dimension sizes(C->rank());
        IndexRange C2inds(C->rank());
        for (size_t ind = 0; ind < Cinds.size(); ind++)
        {
            sizes[ind] = Cinds[ind][1] - Cinds[ind][0];
            C2inds[ind] = {0L, (size_t)sizes[ind]};
            numel *= sizes[ind];
        }

        shared_ptr<CoreTensorImpl> C2(new CoreTensorImpl("C2", sizes));
        double *C2p = C2->data().data();

        std::vector<long_int> Aidx(numel);
        std::vector<size_t> Astrides(A->rank());
        Astrides[0] = 1L;
        for (size_t ind = 1L; ind < A->rank(); ind++)
        {
            Astrides[ind] = Astrides[ind - 1] * A->dim(ind - 1);
        }

        for (size_t ind = 0L; ind < numel; ind++)
        {
            size_t num = ind;
            size_t Aoff = 0L;
            for (int dim = ((int)A->rank()) - 1; dim >= 0; dim--)
            {
                size_t val = num % sizes[dim]; // value of the dim-th index
                num /= sizes[dim];
                Aoff += (Ainds[dim][0] + val) * Astrides[dim];
            }
            Aidx[ind] = Aoff;
        }

        (A->cyclops())->read(numel, 1.0, 0.0, Aidx.data(), C2p);

        C->slice(C2.get(), Cinds, C2inds, alpha, beta);
    }

    timer::timer_pop();
}

void slice(CyclopsTensorImplPtr C, ConstCoreTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha,
           double beta)
{
    timer::timer_push("slice Core -> Cyclops");

    if (C->rank() == 0)
    {
        double Cv = ((CTF_Scalar *)C->cyclops())->get_val();
        double Av = A->data()[0];
        Cv = alpha * Av + beta * Cv;
        ((CTF_Scalar *)C->cyclops())->set_val(Cv);
    }
    else
    {
        long_int numel = 1L;
        Dimension sizes(C->rank());
        IndexRange A2inds(C->rank());
        for (size_t ind = 0; ind < Cinds.size(); ind++)
        {
            sizes[ind] = Cinds[ind][1] - Cinds[ind][0];
            A2inds[ind] = {0L, (size_t)sizes[ind]};
            numel *= sizes[ind];
        }

        shared_ptr<CoreTensorImpl> A2(new CoreTensorImpl("A2", sizes));
        A2->slice(A, A2inds, Ainds, 1.0, 0.0);
        double *A2p = A2->data().data();

        std::vector<long_int> Cidx(numel);
        std::vector<size_t> Cstrides(C->rank());
        Cstrides[0] = 1L;
        for (size_t ind = 1L; ind < C->rank(); ind++)
        {
            Cstrides[ind] = Cstrides[ind - 1] * C->dim(ind - 1);
        }

        for (size_t ind = 0L; ind < numel; ind++)
        {
            size_t num = ind;
            size_t Coff = 0L;
            for (int dim = ((int)C->rank()) - 1; dim >= 0; dim--)
            {
                size_t val = num % sizes[dim]; // value of the dim-th index
                num /= sizes[dim];
                Coff += (Cinds[dim][0] + val) * Cstrides[dim];
            }
            Cidx[ind] = Coff;
        }

        (C->cyclops())->write(numel, alpha, beta, Cidx.data(), A2p);
    }

    timer::timer_pop();
}

void slice(CyclopsTensorImplPtr C, ConstCyclopsTensorImplPtr A,
           const IndexRange &Cinds, const IndexRange &Ainds, double alpha,
           double beta)
{
    timer::timer_push("slice Cyclops -> Cyclops");

    CTF_Tensor *tC = C->cyclops();
    CTF_Tensor *tA = A->cyclops();

    std::vector<int> Coffs(Cinds.size());
    std::vector<int> Cends(Cinds.size());
    std::vector<int> Aoffs(Cinds.size());
    std::vector<int> Aends(Cinds.size());

    for (size_t ind = 0L; ind < Cinds.size(); ind++)
    {
        Coffs[ind] = Cinds[ind][0];
        Cends[ind] = Cinds[ind][1];
        Aoffs[ind] = Ainds[ind][0];
        Aends[ind] = Ainds[ind][1];
    }

    tC->slice(Coffs.data(), Cends.data(), beta, *tA, Aoffs.data(), Aends.data(),
              alpha);

    timer::timer_pop();
}

#endif
}
