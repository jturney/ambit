/*
 * @BEGIN LICENSE
 *
 * ambit: C++ library for the implementation of tensor product calculations
 *        through a clean, concise user interface.
 *
 * Copyright (c) 2014-2017 Ambit developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of ambit.
 *
 * Ambit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Ambit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "math.h"

#include <climits>
#include <cmath>
#include <stdexcept>

//#include <FCMangle.h>

#define FC_GLOBAL(lc, UC) lc ## _

// => BLAS level 1 <=
#define F_DSWAP FC_GLOBAL(dswap, DSWAP)
#define F_DAXPY FC_GLOBAL(daxpy, DAXPY)
#define F_DCOPY FC_GLOBAL(dcopy, DCOPY)
#define F_DROT FC_GLOBAL(drot, DROT)
#define F_DSCAL FC_GLOBAL(dscal, DSCAL)
#define F_DDOT FC_GLOBAL(ddot, DDOT)
#define F_DASUM FC_GLOBAL(dasum, DASUM)
#define F_DNRM2 FC_GLOBAL(dnrm2, DNRM2)
#define F_IDAMAX FC_GLOBAL(idamax, IDAMAX)

// => BLAS level 2/3 <=
#define F_DGBMV FC_GLOBAL(dgbmv, DGBMV)
#define F_DGEMM FC_GLOBAL(dgemm, DGEMM)
#define F_DGEMV FC_GLOBAL(dgemv, DGEMV)
#define F_DGER FC_GLOBAL(dger, DGER)
#define F_DSBMV FC_GLOBAL(dsbmv, DSBMV)
#define F_DSPMV FC_GLOBAL(dspmv, DSPMV)
#define F_DSPR FC_GLOBAL(dspr, DSPR)
#define F_DSPR2 FC_GLOBAL(dspr2, DSPR2)
#define F_DSYMM FC_GLOBAL(dsymm, DSYMM)
#define F_DSYMV FC_GLOBAL(dsymv, DSYMV)
#define F_DSYR FC_GLOBAL(dsyr, DSYR)
#define F_DSYR2 FC_GLOBAL(dsyr2, DSYR2)
#define F_DSYR2K FC_GLOBAL(dsyr2k, DSYR2K)
#define F_DSYRK FC_GLOBAL(dsyrk, DSYRK)
#define F_DTBMV FC_GLOBAL(dtbmv, DTBMV)
#define F_DTBSV FC_GLOBAL(dtbsv, DTBSV)
#define F_DTPMV FC_GLOBAL(dtpmv, DTPMV)
#define F_DTPSV FC_GLOBAL(dtpsv, DTPSV)
#define F_DTRMM FC_GLOBAL(dtrmm, DTRMM)
#define F_DTRMV FC_GLOBAL(dtrmv, DTRMV)
#define F_DTRSM FC_GLOBAL(dtrsm, DTRSM)
#define F_DTRSV FC_GLOBAL(dtrsv, DTRSV)

extern "C" {

// => BLAS level 1 <=
extern void F_DSWAP(int *length, double *x, int *incx, double *y, int *inc_y);
extern void F_DAXPY(int *length, double *a, double *x, int *inc_x, double *y,
                    int *inc_y);
extern void F_DCOPY(int *length, double *x, int *inc_x, double *y, int *inc_y);
extern void F_DGEMM(char *transa, char *transb, int *m, int *n, int *k,
                    double *alpha, double *A, int *lda, double *B, int *ldb,
                    double *beta, double *C, int *ldc);
extern void F_DSYMM(char *side, char *uplo, int *m, int *n, double *alpha,
                    double *A, int *lda, double *B, int *ldb, double *beta,
                    double *C, int *ldc);
extern void F_DROT(int *ntot, double *x, int *incx, double *y, int *incy,
                   double *cotheta, double *sintheta);
extern void F_DSCAL(int *n, double *alpha, double *vec, int *inc);
extern void F_DGEMV(char *transa, int *m, int *n, double *alpha, double *A,
                    int *lda, double *X, int *inc_x, double *beta, double *Y,
                    int *inc_y);
extern void F_DSYMV(char *uplo, int *n, double *alpha, double *A, int *lda,
                    double *X, int *inc_x, double *beta, double *Y, int *inc_y);
extern void F_DSPMV(char *uplo, int *n, double *alpha, double *A, double *X,
                    int *inc_x, double *beta, double *Y, int *inc_y);
extern double F_DDOT(int *n, double *x, int *incx, double *y, int *incy);
extern double F_DNRM2(int *n, double *x, int *incx);
extern double F_DASUM(int *n, double *x, int *incx);
extern int F_IDAMAX(int *n, double *x, int *incx);

// => BLAS level 2/3 <=
extern void F_DGBMV(char *, int *, int *, int *, int *, double *, double *,
                    int *, double *, int *, double *, double *, int *);
extern void F_DGEMM(char *, char *, int *, int *, int *, double *, double *,
                    int *, double *, int *, double *, double *, int *);
extern void F_DGEMV(char *, int *, int *, double *, double *, int *, double *,
                    int *, double *, double *, int *);
extern void F_DGER(int *, int *, double *, double *, int *, double *, int *,
                   double *, int *);
extern void F_DSBMV(char *, int *, int *, double *, double *, int *, double *,
                    int *, double *, double *, int *);
extern void F_DSPMV(char *, int *, double *, double *, double *, int *,
                    double *, double *, int *);
extern void F_DSPR(char *, int *, double *, double *, int *, double *);
extern void F_DSPR2(char *, int *, double *, double *, int *, double *, int *,
                    double *);
extern void F_DSYMM(char *, char *, int *, int *, double *, double *, int *,
                    double *, int *, double *, double *, int *);
extern void F_DSYMV(char *, int *, double *, double *, int *, double *, int *,
                    double *, double *, int *);
extern void F_DSYR(char *, int *, double *, double *, int *, double *, int *);
extern void F_DSYR2(char *, int *, double *, double *, int *, double *, int *,
                    double *, int *);
extern void F_DSYR2K(char *, char *, int *, int *, double *, double *, int *,
                     double *, int *, double *, double *, int *);
extern void F_DSYRK(char *, char *, int *, int *, double *, double *, int *,
                    double *, double *, int *);
extern void F_DTBMV(char *, char *, char *, int *, int *, double *, int *,
                    double *, int *);
extern void F_DTBSV(char *, char *, char *, int *, int *, double *, int *,
                    double *, int *);
extern void F_DTPMV(char *, char *, char *, int *, double *, double *, int *);
extern void F_DTPSV(char *, char *, char *, int *, double *, double *, int *);
extern void F_DTRMM(char *, char *, char *, char *, int *, int *, double *,
                    double *, int *, double *, int *);
extern void F_DTRMV(char *, char *, char *, int *, double *, int *, double *,
                    int *);
extern void F_DTRSM(char *, char *, char *, char *, int *, int *, double *,
                    double *, int *, double *, int *);
extern void F_DTRSV(char *, char *, char *, int *, double *, int *, double *,
                    int *);
}

namespace ambit
{

/**
 * Swaps a vector with another vector.
 *
 * @param length Specifies the number of elements in vectors x and y.
 * @param x Array, DIMENSION at least (1 + (n-1)*abs(incx)).
 * @param inc_x Specifies the increment for the elements of x.
 * @param y Array, DIMENSION at least (1 + (n-1)*abs(incy)).
 * @param inc_y Specifies the increment for the elements of y.
 *
 */
void C_DSWAP(unsigned long int length, double *x, int inc_x, double *y,
             int inc_y)
{
    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *x_s = &x[block * inc_x * (unsigned long int)INT_MAX];
        double *y_s = &y[block * inc_y * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        ::F_DSWAP(&length_s, x_s, &inc_x, y_s, &inc_y);
    }
}

/*!
 * This function performs y = a * x + y.
 *
 * Steps every inc_x in x and every inc_y in y (normally both 1).
 *
 * \param length   length of arrays
 * \param a        scalar a to multiply vector x
 * \param x        vector x
 * \param inc_x    how many places to skip to get to next element in x
 * \param y        vector y
 * \param inc_y    how many places to skip to get to next element in y
 *
 */
void C_DAXPY(unsigned long int length, double a, double *x, int inc_x,
             double *y, int inc_y)
{
    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *x_s = &x[block * inc_x * (unsigned long int)INT_MAX];
        double *y_s = &y[block * inc_y * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        ::F_DAXPY(&length_s, &a, x_s, &inc_x, y_s, &inc_y);
    }
}

/*!
 * This function copies x into y.
 *
 * Steps every inc_x in x and every inc_y in y (normally both 1).
 *
 * \param length  = length of array
 * \param x       = vector x
 * \param inc_x   = how many places to skip to get to next element in x
 * \param y       = vector y
 * \param inc_y   = how many places to skip to get to next element in y
 *
 */
void C_DCOPY(unsigned long int length, double *x, int inc_x, double *y,
             int inc_y)
{
    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *x_s = &x[block * inc_x * (unsigned long int)INT_MAX];
        double *y_s = &y[block * inc_y * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        ::F_DCOPY(&length_s, x_s, &inc_x, y_s, &inc_y);
    }
}

/*!
 * This function scales a vector by a real scalar.
 *
 * \param length length of array
 * \param alpha  scale factor
 * \param vec    vector to scale
 * \param inc    how many places to skip to get to next element in vec
 *
 */
void C_DSCAL(unsigned long int length, double alpha, double *vec, int inc)
{
    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *vec_s = &vec[block * inc * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        ::F_DSCAL(&length_s, &alpha, vec_s, &inc);
    }
}

/*!
 *Calculates a plane Givens rotation for vectors x, y and
 * angle theta.  x = x*cos + y*sin, y = -x*sin + y*cos.
 *
 * \param x      vector x
 * \param y      vector Y
 * \param length length of x,y
 * \param inc_x  how many places to skip to get to the next element of x
 * \param inc_y  how many places to skip to get to the next element of y
 *
 */
void C_DROT(unsigned long int length, double *x, int inc_x, double *y,
            int inc_y, double costheta, double sintheta)
{
    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *x_s = &x[block * inc_x * (unsigned long int)INT_MAX];
        double *y_s = &y[block * inc_y * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        ::F_DROT(&length_s, x_s, &inc_x, y_s, &inc_y, &costheta, &sintheta);
    }
}

/*!
 * This function returns the dot product of two vectors, x and y.
 *
 * \param length Number of elements in x and y.
 * \param x      A pointer to the beginning of the data in x.
 *               Must be of at least length (1+(N-1)*abs(inc_x).
 * \param inc_x  how many places to skip to get to next element in x
 * \param y      A pointer to the beginning of the data in y.
 * \param inc_y  how many places to skip to get to next element in y
 *
 * @return the dot product
 *
 */

double C_DDOT(unsigned long int length, double *x, int inc_x, double *y,
              int inc_y)
{
    if (length == 0)
        return 0.0;

    double reg = 0.0;

    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *x_s = &x[block * inc_x * (unsigned long int)INT_MAX];
        double *y_s = &y[block * inc_y * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        reg += ::F_DDOT(&length_s, x_s, &inc_x, y_s, &inc_y);
    }

    return reg;
}
/*!
 * This function returns the square of the norm of this vector.
 *
 * \param length Number of elements in x.
 * \param x      A pointer to the beginning of the data in x.
 *               Must be of at least length (1+(N-1)*abs(inc_x).
 * \param inc_x  how many places to skip to get to next element in x
 *
 * @return the norm squared product
 *
 */

double C_DNRM2(unsigned long int length, double *x, int inc_x)
{
    if (length == 0)
        return 0.0;

    double reg = 0.0;

    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *x_s = &x[block * inc_x * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        reg += ::F_DNRM2(&length_s, x_s, &inc_x);
    }

    return reg;
}
/*!
 * This function returns the sum of the absolute value of this vector.
 *
 * \param length Number of elements in x.
 * \param x      A pointer to the beginning of the data in x.
 *               Must be of at least length (1+(N-1)*abs(inc_x).
 * \param inc_x  how many places to skip to get to next element in x
 *
 * @return the sum of the absolute value
 *
 */

double C_DASUM(unsigned long int length, double *x, int inc_x)
{
    if (length == 0)
        return 0.0;

    double reg = 0.0;

    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *x_s = &x[block * inc_x * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        reg += ::F_DASUM(&length_s, x_s, &inc_x);
    }

    return reg;
}

/*!
 * This function returns the index of the largest absolute value compoment of
 * this vector.
 *
 * \param length Number of elements in x.
 * \param x      A pointer to the beginning of the data in x.
 *               Must be of at least length (1+(N-1)*abs(inc_x).
 * \param inc_x  how many places to skip to get to next element in x
 *
 * @return the index of the largest absolute value
 *
 */

unsigned long int C_IDAMAX(unsigned long int length, double *x, int inc_x)
{
    if (length == 0)
        return 0L;

    unsigned long int reg = 0L;
    unsigned long int reg2 = 0L;

    int big_blocks = (int)(length / INT_MAX);
    int small_size = (int)(length % INT_MAX);
    for (int block = 0; block <= big_blocks; block++)
    {
        double *x_s = &x[block * inc_x * (unsigned long int)INT_MAX];
        signed int length_s = (block == big_blocks) ? small_size : INT_MAX;
        reg2 = ::F_IDAMAX(&length_s, x_s, &inc_x) +
               block * inc_x * (unsigned long int)INT_MAX;
        if (fabs(x[reg]) > fabs(x[reg2]))
            reg = reg2;
    }

    return reg;
}

/**
 *  Purpose
 *  =======
 *
 *  DGBMV  performs one of the matrix-vector operations
 *
 *     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are vectors and A is an
 *  m by n band matrix, with kl sub-diagonals and ku super-diagonals.
 *
 *  Arguments
 *  ==========
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the operation to be performed as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
 *
 *              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
 *
 *              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry, M specifies the number of rows of the matrix A.
 *           M must be at least zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the number of columns of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  KL     - INTEGER.
 *           On entry, KL specifies the number of sub-diagonals of the
 *           matrix A. KL must satisfy  0 .le. KL.
 *           Unchanged on exit.
 *
 *  KU     - INTEGER.
 *           On entry, KU specifies the number of super-diagonals of the
 *           matrix A. KU must satisfy  0 .le. KU.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry, the leading ( kl + ku + 1 ) by n part of the
 *           array A must contain the matrix of coefficients, supplied
 *           column by column, with the leading diagonal of the matrix in
 *           row ( ku + 1 ) of the array, the first super-diagonal
 *           starting at position 2 in row ku, the first sub-diagonal
 *           starting at position 1 in row ( ku + 2 ), and so on.
 *           Elements in the array A that do not correspond to elements
 *           in the band matrix (such as the top left ku by ku triangle)
 *           are not referenced.
 *           The following program segment will transfer a band matrix
 *           from conventional full matrix storage to band storage:
 *
 *                 DO 20, J = 1, N
 *                    K = KU + 1 - J
 *                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL )
 *                       A( K + I, J ) = matrix( I, J )
 *              10    CONTINUE
 *              20 CONTINUE
 *
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           ( kl + ku + 1 ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of DIMENSION at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
 *           and at least
 *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
 *           Before entry, the incremented array X must contain the
 *           vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta. When BETA is
 *           supplied as zero then Y need not be set on input.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of DIMENSION at least
 *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
 *           and at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
 *           Before entry, the incremented array Y must contain the
 *           vector y. On exit, Y is overwritten by the updated vector y.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *     .. Parameters ..
 **/
void C_DGBMV(char trans, int m, int n, int kl, int ku, double alpha, double *a,
             int lda, double *x, int incx, double beta, double *y, int incy)
{
    if (m == 0 || n == 0)
        return;
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DGBMV trans argument is invalid.");
    ::F_DGBMV(&trans, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
              &incy);
}

/**
 *  Purpose
 *  =======
 *
 *  DGEMM  performs one of the matrix-matrix operations
 *
 *     C := alpha*op( A )*op( B ) + beta*C,
 *
 *  where  op( X ) is one of
 *
 *     op( X ) = X   or   op( X ) = X',
 *
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *
 *  Arguments
 *  ==========
 *
 *  TRANSA - CHARACTER*1.
 *           On entry, TRANSA specifies the form of op( A ) to be used in
 *           the matrix multiplication as follows:
 *
 *              TRANSA = 'N' or 'n',  op( A ) = A.
 *
 *              TRANSA = 'T' or 't',  op( A ) = A'.
 *
 *              TRANSA = 'C' or 'c',  op( A ) = A'.
 *
 *           Unchanged on exit.
 *
 *  TRANSB - CHARACTER*1.
 *           On entry, TRANSB specifies the form of op( B ) to be used in
 *           the matrix multiplication as follows:
 *
 *              TRANSB = 'N' or 'n',  op( B ) = B.
 *
 *              TRANSB = 'T' or 't',  op( B ) = B'.
 *
 *              TRANSB = 'C' or 'c',  op( B ) = B'.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry,  M  specifies  the number  of rows  of the  matrix
 *           op( A )  and of the  matrix  C.  M  must  be at least  zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry,  N  specifies the number  of columns of the matrix
 *           op( B ) and the number of columns of the matrix C. N must be
 *           at least zero.
 *           Unchanged on exit.
 *
 *  K      - INTEGER.
 *           On entry,  K  specifies  the number of columns of the matrix
 *           op( A ) and the number of rows of the matrix op( B ). K must
 *           be at least  zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
 *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
 *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
 *           part of the array  A  must contain the matrix  A,  otherwise
 *           the leading  k by m  part of the array  A  must contain  the
 *           matrix A.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
 *           LDA must be at least  max( 1, m ), otherwise  LDA must be at
 *           least  max( 1, k ).
 *           Unchanged on exit.
 *
 *  B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
 *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
 *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
 *           part of the array  B  must contain the matrix  B,  otherwise
 *           the leading  n by k  part of the array  B  must contain  the
 *           matrix B.
 *           Unchanged on exit.
 *
 *  LDB    - INTEGER.
 *           On entry, LDB specifies the first dimension of B as declared
 *           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
 *           LDB must be at least  max( 1, k ), otherwise  LDB must be at
 *           least  max( 1, n ).
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
 *           supplied as zero then C need not be set on input.
 *           Unchanged on exit.
 *
 *  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
 *           Before entry, the leading  m by n  part of the array  C must
 *           contain the matrix  C,  except when  beta  is zero, in which
 *           case C need not be set on entry.
 *           On exit, the array  C  is overwritten by the  m by n  matrix
 *           ( alpha*op( A )*op( B ) + beta*C ).
 *
 *  LDC    - INTEGER.
 *           On entry, LDC specifies the first dimension of C as declared
 *           in  the  calling  (sub)  program.   LDC  must  be  at  least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *
 *  Level 3 Blas routine.
 *
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 *
 *
 *     .. External Functions ..
 **/
void C_DGEMM(char transa, char transb, int m, int n, int k, double alpha,
             double *a, int lda, double *b, int ldb, double beta, double *c,
             int ldc)
{
    if (m == 0 || n == 0 || k == 0)
        return;
    ::F_DGEMM(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c,
              &ldc);
}

/**
 *  Purpose
 *  =======
 *
 *  DGEMV  performs one of the matrix-vector operations
 *
 *     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are vectors and A is an
 *  m by n matrix.
 *
 *  Arguments
 *  ==========
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the operation to be performed as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
 *
 *              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
 *
 *              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry, M specifies the number of rows of the matrix A.
 *           M must be at least zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the number of columns of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry, the leading m by n part of the array A must
 *           contain the matrix of coefficients.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of DIMENSION at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
 *           and at least
 *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
 *           Before entry, the incremented array X must contain the
 *           vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta. When BETA is
 *           supplied as zero then Y need not be set on input.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of DIMENSION at least
 *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
 *           and at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
 *           Before entry with BETA non-zero, the incremented array Y
 *           must contain the vector y. On exit, Y is overwritten by the
 *           updated vector y.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DGEMV(char trans, int m, int n, double alpha, double *a, int lda,
             double *x, int incx, double beta, double *y, int incy)
{
    if (m == 0 || n == 0)
        return;
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DGEMV trans argument is invalid.");
    ::F_DGEMV(&trans, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

/**
 *  Purpose
 *  =======
 *
 *  DGER   performs the rank 1 operation
 *
 *     A := alpha*x*y' + A,
 *
 *  where alpha is a scalar, x is an m element vector, y is an n element
 *  vector and A is an m by n matrix.
 *
 *  Arguments
 *  ==========
 *
 *  M      - INTEGER.
 *           On entry, M specifies the number of rows of the matrix A.
 *           M must be at least zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the number of columns of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( m - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the m
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ).
 *           Before entry, the incremented array Y must contain the n
 *           element vector y.
 *           Unchanged on exit.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry, the leading m by n part of the array A must
 *           contain the matrix of coefficients. On exit, A is
 *           overwritten by the updated matrix.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DGER(int m, int n, double alpha, double *x, int incx, double *y,
            int incy, double *a, int lda)
{
    if (m == 0 || n == 0)
        return;
    ::F_DGER(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
}

/**
 *  Purpose
 *  =======
 *
 *  DSBMV  performs the matrix-vector  operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n symmetric band matrix, with k super-diagonals.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the band matrix A is being supplied as
 *           follows:
 *
 *              UPLO = 'U' or 'u'   The upper triangular part of A is
 *                                  being supplied.
 *
 *              UPLO = 'L' or 'l'   The lower triangular part of A is
 *                                  being supplied.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  K      - INTEGER.
 *           On entry, K specifies the number of super-diagonals of the
 *           matrix A. K must satisfy  0 .le. K.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
 *           by n part of the array A must contain the upper triangular
 *           band part of the symmetric matrix, supplied column by
 *           column, with the leading diagonal of the matrix in row
 *           ( k + 1 ) of the array, the first super-diagonal starting at
 *           position 2 in row k, and so on. The top left k by k triangle
 *           of the array A is not referenced.
 *           The following program segment will transfer the upper
 *           triangular part of a symmetric band matrix from conventional
 *           full matrix storage to band storage:
 *
 *                 DO 20, J = 1, N
 *                    M = K + 1 - J
 *                    DO 10, I = MAX( 1, J - K ), J
 *                       A( M + I, J ) = matrix( I, J )
 *              10    CONTINUE
 *              20 CONTINUE
 *
 *           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 )
 *           by n part of the array A must contain the lower triangular
 *           band part of the symmetric matrix, supplied column by
 *           column, with the leading diagonal of the matrix in row 1 of
 *           the array, the first sub-diagonal starting at position 1 in
 *           row 2, and so on. The bottom right k by k triangle of the
 *           array A is not referenced.
 *           The following program segment will transfer the lower
 *           triangular part of a symmetric band matrix from conventional
 *           full matrix storage to band storage:
 *
 *                 DO 20, J = 1, N
 *                    M = 1 - J
 *                    DO 10, I = J, MIN( N, J + K )
 *                       A( M + I, J ) = matrix( I, J )
 *              10    CONTINUE
 *              20 CONTINUE
 *
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           ( k + 1 ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of DIMENSION at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the
 *           vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of DIMENSION at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ).
 *           Before entry, the incremented array Y must contain the
 *           vector y. On exit, Y is overwritten by the updated vector y.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DSBMV(char uplo, int n, int k, double alpha, double *a, int lda,
             double *x, int incx, double beta, double *y, int incy)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSBMV uplo argument is invalid.");
    ::F_DSBMV(&uplo, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

/**
 *  Purpose
 *  =======
 *
 *  DSPMV  performs the matrix-vector operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n symmetric matrix, supplied in packed form.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the matrix A is supplied in the packed
 *           array AP as follows:
 *
 *              UPLO = 'U' or 'u'   The upper triangular part of A is
 *                                  supplied in AP.
 *
 *              UPLO = 'L' or 'l'   The lower triangular part of A is
 *                                  supplied in AP.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  AP     - DOUBLE PRECISION array of DIMENSION at least
 *           ( ( n*( n + 1 ) )/2 ).
 *           Before entry with UPLO = 'U' or 'u', the array AP must
 *           contain the upper triangular part of the symmetric matrix
 *           packed sequentially, column by column, so that AP( 1 )
 *           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 )
 *           and a( 2, 2 ) respectively, and so on.
 *           Before entry with UPLO = 'L' or 'l', the array AP must
 *           contain the lower triangular part of the symmetric matrix
 *           packed sequentially, column by column, so that AP( 1 )
 *           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 )
 *           and a( 3, 1 ) respectively, and so on.
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta. When BETA is
 *           supplied as zero then Y need not be set on input.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ).
 *           Before entry, the incremented array Y must contain the n
 *           element vector y. On exit, Y is overwritten by the updated
 *           vector y.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DSPMV(char uplo, int n, double alpha, double *ap, double *x, int incx,
             double beta, double *y, int incy)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSPMV uplo argument is invalid.");
    ::F_DSPMV(&uplo, &n, &alpha, ap, x, &incx, &beta, y, &incy);
}

/**
 *  Purpose
 *  =======
 *
 *  DSPR    performs the symmetric rank 1 operation
 *
 *     A := alpha*x*x' + A,
 *
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n symmetric matrix, supplied in packed form.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the matrix A is supplied in the packed
 *           array AP as follows:
 *
 *              UPLO = 'U' or 'u'   The upper triangular part of A is
 *                                  supplied in AP.
 *
 *              UPLO = 'L' or 'l'   The lower triangular part of A is
 *                                  supplied in AP.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  AP     - DOUBLE PRECISION array of DIMENSION at least
 *           ( ( n*( n + 1 ) )/2 ).
 *           Before entry with  UPLO = 'U' or 'u', the array AP must
 *           contain the upper triangular part of the symmetric matrix
 *           packed sequentially, column by column, so that AP( 1 )
 *           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 )
 *           and a( 2, 2 ) respectively, and so on. On exit, the array
 *           AP is overwritten by the upper triangular part of the
 *           updated matrix.
 *           Before entry with UPLO = 'L' or 'l', the array AP must
 *           contain the lower triangular part of the symmetric matrix
 *           packed sequentially, column by column, so that AP( 1 )
 *           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 )
 *           and a( 3, 1 ) respectively, and so on. On exit, the array
 *           AP is overwritten by the lower triangular part of the
 *           updated matrix.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DSPR(char uplo, int n, double alpha, double *x, int incx, double *ap)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSPR uplo argument is invalid.");
    ::F_DSPR(&uplo, &n, &alpha, x, &incx, ap);
}

/**
 *  Purpose
 *  =======
 *
 *  DSPR2  performs the symmetric rank 2 operation
 *
 *     A := alpha*x*y' + alpha*y*x' + A,
 *
 *  where alpha is a scalar, x and y are n element vectors and A is an
 *  n by n symmetric matrix, supplied in packed form.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the matrix A is supplied in the packed
 *           array AP as follows:
 *
 *              UPLO = 'U' or 'u'   The upper triangular part of A is
 *                                  supplied in AP.
 *
 *              UPLO = 'L' or 'l'   The lower triangular part of A is
 *                                  supplied in AP.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ).
 *           Before entry, the incremented array Y must contain the n
 *           element vector y.
 *           Unchanged on exit.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *  AP     - DOUBLE PRECISION array of DIMENSION at least
 *           ( ( n*( n + 1 ) )/2 ).
 *           Before entry with  UPLO = 'U' or 'u', the array AP must
 *           contain the upper triangular part of the symmetric matrix
 *           packed sequentially, column by column, so that AP( 1 )
 *           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 )
 *           and a( 2, 2 ) respectively, and so on. On exit, the array
 *           AP is overwritten by the upper triangular part of the
 *           updated matrix.
 *           Before entry with UPLO = 'L' or 'l', the array AP must
 *           contain the lower triangular part of the symmetric matrix
 *           packed sequentially, column by column, so that AP( 1 )
 *           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 )
 *           and a( 3, 1 ) respectively, and so on. On exit, the array
 *           AP is overwritten by the lower triangular part of the
 *           updated matrix.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DSPR2(char uplo, int n, double alpha, double *x, int incx, double *y,
             int incy, double *ap)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSPR2 uplo argument is invalid.");
    ::F_DSPR2(&uplo, &n, &alpha, x, &incx, y, &incy, ap);
}

/**
 *  Purpose
 *  =======
 *
 *  DSYMM  performs one of the matrix-matrix operations
 *
 *     C := alpha*A*B + beta*C,
 *
 *  or
 *
 *     C := alpha*B*A + beta*C,
 *
 *  where alpha and beta are scalars,  A is a symmetric matrix and  B and
 *  C are  m by n matrices.
 *
 *  Arguments
 *  ==========
 *
 *  SIDE   - CHARACTER*1.
 *           On entry,  SIDE  specifies whether  the  symmetric matrix  A
 *           appears on the  left or right  in the  operation as follows:
 *
 *              SIDE = 'L' or 'l'   C := alpha*A*B + beta*C,
 *
 *              SIDE = 'R' or 'r'   C := alpha*B*A + beta*C,
 *
 *           Unchanged on exit.
 *
 *  UPLO   - CHARACTER*1.
 *           On  entry,   UPLO  specifies  whether  the  upper  or  lower
 *           triangular  part  of  the  symmetric  matrix   A  is  to  be
 *           referenced as follows:
 *
 *              UPLO = 'U' or 'u'   Only the upper triangular part of the
 *                                  symmetric matrix is to be referenced.
 *
 *              UPLO = 'L' or 'l'   Only the lower triangular part of the
 *                                  symmetric matrix is to be referenced.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry,  M  specifies the number of rows of the matrix  C.
 *           M  must be at least zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the number of columns of the matrix C.
 *           N  must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
 *           m  when  SIDE = 'L' or 'l'  and is  n otherwise.
 *           Before entry  with  SIDE = 'L' or 'l',  the  m by m  part of
 *           the array  A  must contain the  symmetric matrix,  such that
 *           when  UPLO = 'U' or 'u', the leading m by m upper triangular
 *           part of the array  A  must contain the upper triangular part
 *           of the  symmetric matrix and the  strictly  lower triangular
 *           part of  A  is not referenced,  and when  UPLO = 'L' or 'l',
 *           the leading  m by m  lower triangular part  of the  array  A
 *           must  contain  the  lower triangular part  of the  symmetric
 *           matrix and the  strictly upper triangular part of  A  is not
 *           referenced.
 *           Before entry  with  SIDE = 'R' or 'r',  the  n by n  part of
 *           the array  A  must contain the  symmetric matrix,  such that
 *           when  UPLO = 'U' or 'u', the leading n by n upper triangular
 *           part of the array  A  must contain the upper triangular part
 *           of the  symmetric matrix and the  strictly  lower triangular
 *           part of  A  is not referenced,  and when  UPLO = 'L' or 'l',
 *           the leading  n by n  lower triangular part  of the  array  A
 *           must  contain  the  lower triangular part  of the  symmetric
 *           matrix and the  strictly upper triangular part of  A  is not
 *           referenced.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
 *           LDA must be at least  max( 1, m ), otherwise  LDA must be at
 *           least  max( 1, n ).
 *           Unchanged on exit.
 *
 *  B      - DOUBLE PRECISION array of DIMENSION ( LDB, n ).
 *           Before entry, the leading  m by n part of the array  B  must
 *           contain the matrix B.
 *           Unchanged on exit.
 *
 *  LDB    - INTEGER.
 *           On entry, LDB specifies the first dimension of B as declared
 *           in  the  calling  (sub)  program.   LDB  must  be  at  least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
 *           supplied as zero then C need not be set on input.
 *           Unchanged on exit.
 *
 *  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
 *           Before entry, the leading  m by n  part of the array  C must
 *           contain the matrix  C,  except when  beta  is zero, in which
 *           case C need not be set on entry.
 *           On exit, the array  C  is overwritten by the  m by n updated
 *           matrix.
 *
 *  LDC    - INTEGER.
 *           On entry, LDC specifies the first dimension of C as declared
 *           in  the  calling  (sub)  program.   LDC  must  be  at  least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *
 *  Level 3 Blas routine.
 *
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 *
 *
 *     .. External Functions ..
 **/
void C_DSYMM(char side, char uplo, int m, int n, double alpha, double *a,
             int lda, double *b, int ldb, double beta, double *c, int ldc)
{
    if (m == 0 || n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSYMM uplo argument is invalid.");
    if (side == 'L' || side == 'L')
        side = 'R';
    else if (side == 'R' || side == 'r')
        side = 'L';
    else
        throw std::invalid_argument("C_DSYMM side argument is invalid.");
    ::F_DSYMM(&side, &uplo, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

/**
 *  Purpose
 *  =======
 *
 *  DSYMV  performs the matrix-vector  operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n symmetric matrix.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the array A is to be referenced as
 *           follows:
 *
 *              UPLO = 'U' or 'u'   Only the upper triangular part of A
 *                                  is to be referenced.
 *
 *              UPLO = 'L' or 'l'   Only the lower triangular part of A
 *                                  is to be referenced.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
 *           upper triangular part of the array A must contain the upper
 *           triangular part of the symmetric matrix and the strictly
 *           lower triangular part of A is not referenced.
 *           Before entry with UPLO = 'L' or 'l', the leading n by n
 *           lower triangular part of the array A must contain the lower
 *           triangular part of the symmetric matrix and the strictly
 *           upper triangular part of A is not referenced.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta. When BETA is
 *           supplied as zero then Y need not be set on input.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ).
 *           Before entry, the incremented array Y must contain the n
 *           element vector y. On exit, Y is overwritten by the updated
 *           vector y.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DSYMV(char uplo, int n, double alpha, double *a, int lda, double *x,
             int incx, double beta, double *y, int incy)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSYMV uplo argument is invalid.");
    ::F_DSYMV(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

/**
 *  Purpose
 *  =======
 *
 *  DSYR   performs the symmetric rank 1 operation
 *
 *     A := alpha*x*x' + A,
 *
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n symmetric matrix.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the array A is to be referenced as
 *           follows:
 *
 *              UPLO = 'U' or 'u'   Only the upper triangular part of A
 *                                  is to be referenced.
 *
 *              UPLO = 'L' or 'l'   Only the lower triangular part of A
 *                                  is to be referenced.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
 *           upper triangular part of the array A must contain the upper
 *           triangular part of the symmetric matrix and the strictly
 *           lower triangular part of A is not referenced. On exit, the
 *           upper triangular part of the array A is overwritten by the
 *           upper triangular part of the updated matrix.
 *           Before entry with UPLO = 'L' or 'l', the leading n by n
 *           lower triangular part of the array A must contain the lower
 *           triangular part of the symmetric matrix and the strictly
 *           upper triangular part of A is not referenced. On exit, the
 *           lower triangular part of the array A is overwritten by the
 *           lower triangular part of the updated matrix.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DSYR(char uplo, int n, double alpha, double *x, int incx, double *a,
            int lda)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSYR uplo argument is invalid.");
    ::F_DSYR(&uplo, &n, &alpha, x, &incx, a, &lda);
}

/**
 *  Purpose
 *  =======
 *
 *  DSYR2  performs the symmetric rank 2 operation
 *
 *     A := alpha*x*y' + alpha*y*x' + A,
 *
 *  where alpha is a scalar, x and y are n element vectors and A is an n
 *  by n symmetric matrix.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the array A is to be referenced as
 *           follows:
 *
 *              UPLO = 'U' or 'u'   Only the upper triangular part of A
 *                                  is to be referenced.
 *
 *              UPLO = 'L' or 'l'   Only the lower triangular part of A
 *                                  is to be referenced.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ).
 *           Before entry, the incremented array Y must contain the n
 *           element vector y.
 *           Unchanged on exit.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
 *           upper triangular part of the array A must contain the upper
 *           triangular part of the symmetric matrix and the strictly
 *           lower triangular part of A is not referenced. On exit, the
 *           upper triangular part of the array A is overwritten by the
 *           upper triangular part of the updated matrix.
 *           Before entry with UPLO = 'L' or 'l', the leading n by n
 *           lower triangular part of the array A must contain the lower
 *           triangular part of the symmetric matrix and the strictly
 *           upper triangular part of A is not referenced. On exit, the
 *           lower triangular part of the array A is overwritten by the
 *           lower triangular part of the updated matrix.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DSYR2(char uplo, int n, double alpha, double *x, int incx, double *y,
             int incy, double *a, int lda)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSYR2 uplo argument is invalid.");
    ::F_DSYR2(&uplo, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

/**
 *  Purpose
 *  =======
 *
 *  DSYR2K  performs one of the symmetric rank 2k operations
 *
 *     C := alpha*A*B' + alpha*B*A' + beta*C,
 *
 *  or
 *
 *     C := alpha*A'*B + alpha*B'*A + beta*C,
 *
 *  where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
 *  and  A and B  are  n by k  matrices  in the  first  case  and  k by n
 *  matrices in the second case.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On  entry,   UPLO  specifies  whether  the  upper  or  lower
 *           triangular  part  of the  array  C  is to be  referenced  as
 *           follows:
 *
 *              UPLO = 'U' or 'u'   Only the  upper triangular part of  C
 *                                  is to be referenced.
 *
 *              UPLO = 'L' or 'l'   Only the  lower triangular part of  C
 *                                  is to be referenced.
 *
 *           Unchanged on exit.
 *
 *  TRANS  - CHARACTER*1.
 *           On entry,  TRANS  specifies the operation to be performed as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   C := alpha*A*B' + alpha*B*A' +
 *                                        beta*C.
 *
 *              TRANS = 'T' or 't'   C := alpha*A'*B + alpha*B'*A +
 *                                        beta*C.
 *
 *              TRANS = 'C' or 'c'   C := alpha*A'*B + alpha*B'*A +
 *                                        beta*C.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry,  N specifies the order of the matrix C.  N must be
 *           at least zero.
 *           Unchanged on exit.
 *
 *  K      - INTEGER.
 *           On entry with  TRANS = 'N' or 'n',  K  specifies  the number
 *           of  columns  of the  matrices  A and B,  and on  entry  with
 *           TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
 *           of rows of the matrices  A and B.  K must be at least  zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
 *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
 *           Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
 *           part of the array  A  must contain the matrix  A,  otherwise
 *           the leading  k by n  part of the array  A  must contain  the
 *           matrix A.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
 *           then  LDA must be at least  max( 1, n ), otherwise  LDA must
 *           be at least  max( 1, k ).
 *           Unchanged on exit.
 *
 *  B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
 *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
 *           Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
 *           part of the array  B  must contain the matrix  B,  otherwise
 *           the leading  k by n  part of the array  B  must contain  the
 *           matrix B.
 *           Unchanged on exit.
 *
 *  LDB    - INTEGER.
 *           On entry, LDB specifies the first dimension of B as declared
 *           in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
 *           then  LDB must be at least  max( 1, n ), otherwise  LDB must
 *           be at least  max( 1, k ).
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta.
 *           Unchanged on exit.
 *
 *  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
 *           Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
 *           upper triangular part of the array C must contain the upper
 *           triangular part  of the  symmetric matrix  and the strictly
 *           lower triangular part of C is not referenced.  On exit, the
 *           upper triangular part of the array  C is overwritten by the
 *           upper triangular part of the updated matrix.
 *           Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
 *           lower triangular part of the array C must contain the lower
 *           triangular part  of the  symmetric matrix  and the strictly
 *           upper triangular part of C is not referenced.  On exit, the
 *           lower triangular part of the array  C is overwritten by the
 *           lower triangular part of the updated matrix.
 *
 *  LDC    - INTEGER.
 *           On entry, LDC specifies the first dimension of C as declared
 *           in  the  calling  (sub)  program.   LDC  must  be  at  least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *
 *  Level 3 Blas routine.
 *
 *
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 *
 *
 *     .. External Functions ..
 **/
void C_DSYR2K(char uplo, char trans, int n, int k, double alpha, double *a,
              int lda, double *b, int ldb, double beta, double *c, int ldc)
{
    if (n == 0 || k == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSYR2K uplo argument is invalid.");
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DSYR2K trans argument is invalid.");

    ::F_DSYR2K(&uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

/**
 *  Purpose
 *  =======
 *
 *  DSYRK  performs one of the symmetric rank k operations
 *
 *     C := alpha*A*A' + beta*C,
 *
 *  or
 *
 *     C := alpha*A'*A + beta*C,
 *
 *  where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
 *  and  A  is an  n by k  matrix in the first case and a  k by n  matrix
 *  in the second case.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On  entry,   UPLO  specifies  whether  the  upper  or  lower
 *           triangular  part  of the  array  C  is to be  referenced  as
 *           follows:
 *
 *              UPLO = 'U' or 'u'   Only the  upper triangular part of  C
 *                                  is to be referenced.
 *
 *              UPLO = 'L' or 'l'   Only the  lower triangular part of  C
 *                                  is to be referenced.
 *
 *           Unchanged on exit.
 *
 *  TRANS  - CHARACTER*1.
 *           On entry,  TRANS  specifies the operation to be performed as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   C := alpha*A*A' + beta*C.
 *
 *              TRANS = 'T' or 't'   C := alpha*A'*A + beta*C.
 *
 *              TRANS = 'C' or 'c'   C := alpha*A'*A + beta*C.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry,  N specifies the order of the matrix C.  N must be
 *           at least zero.
 *           Unchanged on exit.
 *
 *  K      - INTEGER.
 *           On entry with  TRANS = 'N' or 'n',  K  specifies  the number
 *           of  columns   of  the   matrix   A,   and  on   entry   with
 *           TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
 *           of rows of the matrix  A.  K must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
 *           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
 *           Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
 *           part of the array  A  must contain the matrix  A,  otherwise
 *           the leading  k by n  part of the array  A  must contain  the
 *           matrix A.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
 *           then  LDA must be at least  max( 1, n ), otherwise  LDA must
 *           be at least  max( 1, k ).
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta.
 *           Unchanged on exit.
 *
 *  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
 *           Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
 *           upper triangular part of the array C must contain the upper
 *           triangular part  of the  symmetric matrix  and the strictly
 *           lower triangular part of C is not referenced.  On exit, the
 *           upper triangular part of the array  C is overwritten by the
 *           upper triangular part of the updated matrix.
 *           Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
 *           lower triangular part of the array C must contain the lower
 *           triangular part  of the  symmetric matrix  and the strictly
 *           upper triangular part of C is not referenced.  On exit, the
 *           lower triangular part of the array  C is overwritten by the
 *           lower triangular part of the updated matrix.
 *
 *  LDC    - INTEGER.
 *           On entry, LDC specifies the first dimension of C as declared
 *           in  the  calling  (sub)  program.   LDC  must  be  at  least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *
 *  Level 3 Blas routine.
 *
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 *
 *
 *     .. External Functions ..
 **/
void C_DSYRK(char uplo, char trans, int n, int k, double alpha, double *a,
             int lda, double beta, double *c, int ldc)
{
    if (n == 0 || k == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DSYRK uplo argument is invalid.");
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DSYRK trans argument is invalid.");

    ::F_DSYRK(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/**
 *  Purpose
 *  =======
 *
 *  DTBMV  performs one of the matrix-vector operations
 *
 *     x := A*x,   or   x := A'*x,
 *
 *  where x is an n element vector and  A is an n by n unit, or non-unit,
 *  upper or lower triangular band matrix, with ( k + 1 ) diagonals.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the matrix is an upper or
 *           lower triangular matrix as follows:
 *
 *              UPLO = 'U' or 'u'   A is an upper triangular matrix.
 *
 *              UPLO = 'L' or 'l'   A is a lower triangular matrix.
 *
 *           Unchanged on exit.
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the operation to be performed as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   x := A*x.
 *
 *              TRANS = 'T' or 't'   x := A'*x.
 *
 *              TRANS = 'C' or 'c'   x := A'*x.
 *
 *           Unchanged on exit.
 *
 *  DIAG   - CHARACTER*1.
 *           On entry, DIAG specifies whether or not A is unit
 *           triangular as follows:
 *
 *              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *
 *              DIAG = 'N' or 'n'   A is not assumed to be unit
 *                                  triangular.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  K      - INTEGER.
 *           On entry with UPLO = 'U' or 'u', K specifies the number of
 *           super-diagonals of the matrix A.
 *           On entry with UPLO = 'L' or 'l', K specifies the number of
 *           sub-diagonals of the matrix A.
 *           K must satisfy  0 .le. K.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
 *           by n part of the array A must contain the upper triangular
 *           band part of the matrix of coefficients, supplied column by
 *           column, with the leading diagonal of the matrix in row
 *           ( k + 1 ) of the array, the first super-diagonal starting at
 *           position 2 in row k, and so on. The top left k by k triangle
 *           of the array A is not referenced.
 *           The following program segment will transfer an upper
 *           triangular band matrix from conventional full matrix storage
 *           to band storage:
 *
 *                 DO 20, J = 1, N
 *                    M = K + 1 - J
 *                    DO 10, I = MAX( 1, J - K ), J
 *                       A( M + I, J ) = matrix( I, J )
 *              10    CONTINUE
 *              20 CONTINUE
 *
 *           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 )
 *           by n part of the array A must contain the lower triangular
 *           band part of the matrix of coefficients, supplied column by
 *           column, with the leading diagonal of the matrix in row 1 of
 *           the array, the first sub-diagonal starting at position 1 in
 *           row 2, and so on. The bottom right k by k triangle of the
 *           array A is not referenced.
 *           The following program segment will transfer a lower
 *           triangular band matrix from conventional full matrix storage
 *           to band storage:
 *
 *                 DO 20, J = 1, N
 *                    M = 1 - J
 *                    DO 10, I = J, MIN( N, J + K )
 *                       A( M + I, J ) = matrix( I, J )
 *              10    CONTINUE
 *              20 CONTINUE
 *
 *           Note that when DIAG = 'U' or 'u' the elements of the array A
 *           corresponding to the diagonal elements of the matrix are not
 *           referenced, but are assumed to be unity.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           ( k + 1 ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x. On exit, X is overwritten with the
 *           tranformed vector x.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DTBMV(char uplo, char trans, char diag, int n, int k, double *a, int lda,
             double *x, int incx)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DTBMV uplo argument is invalid.");
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DTBMV trans argument is invalid.");
    ::F_DTBMV(&uplo, &trans, &diag, &n, &k, a, &lda, x, &incx);
}

/**
 *  Purpose
 *  =======
 *
 *  DTBSV  solves one of the systems of equations
 *
 *     A*x = b,   or   A'*x = b,
 *
 *  where b and x are n element vectors and A is an n by n unit, or
 *  non-unit, upper or lower triangular band matrix, with ( k + 1 )
 *  diagonals.
 *
 *  No test for singularity or near-singularity is included in this
 *  routine. Such tests must be performed before calling this routine.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the matrix is an upper or
 *           lower triangular matrix as follows:
 *
 *              UPLO = 'U' or 'u'   A is an upper triangular matrix.
 *
 *              UPLO = 'L' or 'l'   A is a lower triangular matrix.
 *
 *           Unchanged on exit.
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the equations to be solved as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   A*x = b.
 *
 *              TRANS = 'T' or 't'   A'*x = b.
 *
 *              TRANS = 'C' or 'c'   A'*x = b.
 *
 *           Unchanged on exit.
 *
 *  DIAG   - CHARACTER*1.
 *           On entry, DIAG specifies whether or not A is unit
 *           triangular as follows:
 *
 *              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *
 *              DIAG = 'N' or 'n'   A is not assumed to be unit
 *                                  triangular.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  K      - INTEGER.
 *           On entry with UPLO = 'U' or 'u', K specifies the number of
 *           super-diagonals of the matrix A.
 *           On entry with UPLO = 'L' or 'l', K specifies the number of
 *           sub-diagonals of the matrix A.
 *           K must satisfy  0 .le. K.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 )
 *           by n part of the array A must contain the upper triangular
 *           band part of the matrix of coefficients, supplied column by
 *           column, with the leading diagonal of the matrix in row
 *           ( k + 1 ) of the array, the first super-diagonal starting at
 *           position 2 in row k, and so on. The top left k by k triangle
 *           of the array A is not referenced.
 *           The following program segment will transfer an upper
 *           triangular band matrix from conventional full matrix storage
 *           to band storage:
 *
 *                 DO 20, J = 1, N
 *                    M = K + 1 - J
 *                    DO 10, I = MAX( 1, J - K ), J
 *                       A( M + I, J ) = matrix( I, J )
 *              10    CONTINUE
 *              20 CONTINUE
 *
 *           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 )
 *           by n part of the array A must contain the lower triangular
 *           band part of the matrix of coefficients, supplied column by
 *           column, with the leading diagonal of the matrix in row 1 of
 *           the array, the first sub-diagonal starting at position 1 in
 *           row 2, and so on. The bottom right k by k triangle of the
 *           array A is not referenced.
 *           The following program segment will transfer a lower
 *           triangular band matrix from conventional full matrix storage
 *           to band storage:
 *
 *                 DO 20, J = 1, N
 *                    M = 1 - J
 *                    DO 10, I = J, MIN( N, J + K )
 *                       A( M + I, J ) = matrix( I, J )
 *              10    CONTINUE
 *              20 CONTINUE
 *
 *           Note that when DIAG = 'U' or 'u' the elements of the array A
 *           corresponding to the diagonal elements of the matrix are not
 *           referenced, but are assumed to be unity.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           ( k + 1 ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element right-hand side vector b. On exit, X is overwritten
 *           with the solution vector x.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DTBSV(char uplo, char trans, char diag, int n, int k, double *a, int lda,
             double *x, int incx)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DTBSV uplo argument is invalid.");
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DTBSV trans argument is invalid.");
    ::F_DTBSV(&uplo, &trans, &diag, &n, &k, a, &lda, x, &incx);
}

/**
 *  Purpose
 *  =======
 *
 *  DTPMV  performs one of the matrix-vector operations
 *
 *     x := A*x,   or   x := A'*x,
 *
 *  where x is an n element vector and  A is an n by n unit, or non-unit,
 *  upper or lower triangular matrix, supplied in packed form.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the matrix is an upper or
 *           lower triangular matrix as follows:
 *
 *              UPLO = 'U' or 'u'   A is an upper triangular matrix.
 *
 *              UPLO = 'L' or 'l'   A is a lower triangular matrix.
 *
 *           Unchanged on exit.
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the operation to be performed as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   x := A*x.
 *
 *              TRANS = 'T' or 't'   x := A'*x.
 *
 *              TRANS = 'C' or 'c'   x := A'*x.
 *
 *           Unchanged on exit.
 *
 *  DIAG   - CHARACTER*1.
 *           On entry, DIAG specifies whether or not A is unit
 *           triangular as follows:
 *
 *              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *
 *              DIAG = 'N' or 'n'   A is not assumed to be unit
 *                                  triangular.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  AP     - DOUBLE PRECISION array of DIMENSION at least
 *           ( ( n*( n + 1 ) )/2 ).
 *           Before entry with  UPLO = 'U' or 'u', the array AP must
 *           contain the upper triangular matrix packed sequentially,
 *           column by column, so that AP( 1 ) contains a( 1, 1 ),
 *           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 )
 *           respectively, and so on.
 *           Before entry with UPLO = 'L' or 'l', the array AP must
 *           contain the lower triangular matrix packed sequentially,
 *           column by column, so that AP( 1 ) contains a( 1, 1 ),
 *           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 )
 *           respectively, and so on.
 *           Note that when  DIAG = 'U' or 'u', the diagonal elements of
 *           A are not referenced, but are assumed to be unity.
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x. On exit, X is overwritten with the
 *           tranformed vector x.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DTPMV(char uplo, char trans, char diag, int n, double *ap, double *x,
             int incx)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DTPMV uplo argument is invalid.");
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DTPMV trans argument is invalid.");
    ::F_DTPMV(&uplo, &trans, &diag, &n, ap, x, &incx);
}

/**
 *  Purpose
 *  =======
 *
 *  DTPSV  solves one of the systems of equations
 *
 *     A*x = b,   or   A'*x = b,
 *
 *  where b and x are n element vectors and A is an n by n unit, or
 *  non-unit, upper or lower triangular matrix, supplied in packed form.
 *
 *  No test for singularity or near-singularity is included in this
 *  routine. Such tests must be performed before calling this routine.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the matrix is an upper or
 *           lower triangular matrix as follows:
 *
 *              UPLO = 'U' or 'u'   A is an upper triangular matrix.
 *
 *              UPLO = 'L' or 'l'   A is a lower triangular matrix.
 *
 *           Unchanged on exit.
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the equations to be solved as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   A*x = b.
 *
 *              TRANS = 'T' or 't'   A'*x = b.
 *
 *              TRANS = 'C' or 'c'   A'*x = b.
 *
 *           Unchanged on exit.
 *
 *  DIAG   - CHARACTER*1.
 *           On entry, DIAG specifies whether or not A is unit
 *           triangular as follows:
 *
 *              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *
 *              DIAG = 'N' or 'n'   A is not assumed to be unit
 *                                  triangular.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  AP     - DOUBLE PRECISION array of DIMENSION at least
 *           ( ( n*( n + 1 ) )/2 ).
 *           Before entry with  UPLO = 'U' or 'u', the array AP must
 *           contain the upper triangular matrix packed sequentially,
 *           column by column, so that AP( 1 ) contains a( 1, 1 ),
 *           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 )
 *           respectively, and so on.
 *           Before entry with UPLO = 'L' or 'l', the array AP must
 *           contain the lower triangular matrix packed sequentially,
 *           column by column, so that AP( 1 ) contains a( 1, 1 ),
 *           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 )
 *           respectively, and so on.
 *           Note that when  DIAG = 'U' or 'u', the diagonal elements of
 *           A are not referenced, but are assumed to be unity.
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element right-hand side vector b. On exit, X is overwritten
 *           with the solution vector x.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DTPSV(char uplo, char trans, char diag, int n, double *ap, double *x,
             int incx)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DTPSV uplo argument is invalid.");
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DTPSV trans argument is invalid.");
    ::F_DTPSV(&uplo, &trans, &diag, &n, ap, x, &incx);
}

/**
 *  Purpose
 *  =======
 *
 *  DTRMM  performs one of the matrix-matrix operations
 *
 *     B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
 *
 *  where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
 *  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
 *
 *     op( A ) = A   or   op( A ) = A'.
 *
 *  Arguments
 *  ==========
 *
 *  SIDE   - CHARACTER*1.
 *           On entry,  SIDE specifies whether  op( A ) multiplies B from
 *           the left or right as follows:
 *
 *              SIDE = 'L' or 'l'   B := alpha*op( A )*B.
 *
 *              SIDE = 'R' or 'r'   B := alpha*B*op( A ).
 *
 *           Unchanged on exit.
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the matrix A is an upper or
 *           lower triangular matrix as follows:
 *
 *              UPLO = 'U' or 'u'   A is an upper triangular matrix.
 *
 *              UPLO = 'L' or 'l'   A is a lower triangular matrix.
 *
 *           Unchanged on exit.
 *
 *  TRANSA - CHARACTER*1.
 *           On entry, TRANSA specifies the form of op( A ) to be used in
 *           the matrix multiplication as follows:
 *
 *              TRANSA = 'N' or 'n'   op( A ) = A.
 *
 *              TRANSA = 'T' or 't'   op( A ) = A'.
 *
 *              TRANSA = 'C' or 'c'   op( A ) = A'.
 *
 *           Unchanged on exit.
 *
 *  DIAG   - CHARACTER*1.
 *           On entry, DIAG specifies whether or not A is unit triangular
 *           as follows:
 *
 *              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *
 *              DIAG = 'N' or 'n'   A is not assumed to be unit
 *                                  triangular.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry, M specifies the number of rows of B. M must be at
 *           least zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the number of columns of B.  N must be
 *           at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry,  ALPHA specifies the scalar  alpha. When  alpha is
 *           zero then  A is not referenced and  B need not be set before
 *           entry.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, k ), where k is m
 *           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
 *           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
 *           upper triangular part of the array  A must contain the upper
 *           triangular matrix  and the strictly lower triangular part of
 *           A is not referenced.
 *           Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
 *           lower triangular part of the array  A must contain the lower
 *           triangular matrix  and the strictly upper triangular part of
 *           A is not referenced.
 *           Note that when  DIAG = 'U' or 'u',  the diagonal elements of
 *           A  are not referenced either,  but are assumed to be  unity.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
 *           LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
 *           then LDA must be at least max( 1, n ).
 *           Unchanged on exit.
 *
 *  B      - DOUBLE PRECISION array of DIMENSION ( LDB, n ).
 *           Before entry,  the leading  m by n part of the array  B must
 *           contain the matrix  B,  and  on exit  is overwritten  by the
 *           transformed matrix.
 *
 *  LDB    - INTEGER.
 *           On entry, LDB specifies the first dimension of B as declared
 *           in  the  calling  (sub)  program.   LDB  must  be  at  least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *
 *  Level 3 Blas routine.
 *
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 *
 *
 *     .. External Functions ..
 **/
void C_DTRMM(char side, char uplo, char transa, char diag, int m, int n,
             double alpha, double *a, int lda, double *b, int ldb)
{
    if (m == 0 || n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DTRMM uplo argument is invalid.");
    if (side == 'L' || side == 'L')
        side = 'R';
    else if (side == 'R' || side == 'r')
        side = 'L';
    else
        throw std::invalid_argument("C_DTRMM side argument is invalid.");
    ::F_DTRMM(&side, &uplo, &transa, &diag, &n, &m, &alpha, a, &lda, b, &ldb);
}

/**
 *  Purpose
 *  =======
 *
 *  DTRMV  performs one of the matrix-vector operations
 *
 *     x := A*x,   or   x := A'*x,
 *
 *  where x is an n element vector and  A is an n by n unit, or non-unit,
 *  upper or lower triangular matrix.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the matrix is an upper or
 *           lower triangular matrix as follows:
 *
 *              UPLO = 'U' or 'u'   A is an upper triangular matrix.
 *
 *              UPLO = 'L' or 'l'   A is a lower triangular matrix.
 *
 *           Unchanged on exit.
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the operation to be performed as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   x := A*x.
 *
 *              TRANS = 'T' or 't'   x := A'*x.
 *
 *              TRANS = 'C' or 'c'   x := A'*x.
 *
 *           Unchanged on exit.
 *
 *  DIAG   - CHARACTER*1.
 *           On entry, DIAG specifies whether or not A is unit
 *           triangular as follows:
 *
 *              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *
 *              DIAG = 'N' or 'n'   A is not assumed to be unit
 *                                  triangular.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
 *           upper triangular part of the array A must contain the upper
 *           triangular matrix and the strictly lower triangular part of
 *           A is not referenced.
 *           Before entry with UPLO = 'L' or 'l', the leading n by n
 *           lower triangular part of the array A must contain the lower
 *           triangular matrix and the strictly upper triangular part of
 *           A is not referenced.
 *           Note that when  DIAG = 'U' or 'u', the diagonal elements of
 *           A are not referenced either, but are assumed to be unity.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x. On exit, X is overwritten with the
 *           tranformed vector x.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DTRMV(char uplo, char trans, char diag, int n, double *a, int lda,
             double *x, int incx)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DTRMV uplo argument is invalid.");
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DTRMV trans argument is invalid.");
    ::F_DTRMV(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
}

/**
 *  Purpose
 *  =======
 *
 *  DTRSM  solves one of the matrix equations
 *
 *     op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
 *
 *  where alpha is a scalar, X and B are m by n matrices, A is a unit, or
 *  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
 *
 *     op( A ) = A   or   op( A ) = A'.
 *
 *  The matrix X is overwritten on B.
 *
 *  Arguments
 *  ==========
 *
 *  SIDE   - CHARACTER*1.
 *           On entry, SIDE specifies whether op( A ) appears on the left
 *           or right of X as follows:
 *
 *              SIDE = 'L' or 'l'   op( A )*X = alpha*B.
 *
 *              SIDE = 'R' or 'r'   X*op( A ) = alpha*B.
 *
 *           Unchanged on exit.
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the matrix A is an upper or
 *           lower triangular matrix as follows:
 *
 *              UPLO = 'U' or 'u'   A is an upper triangular matrix.
 *
 *              UPLO = 'L' or 'l'   A is a lower triangular matrix.
 *
 *           Unchanged on exit.
 *
 *  TRANSA - CHARACTER*1.
 *           On entry, TRANSA specifies the form of op( A ) to be used in
 *           the matrix multiplication as follows:
 *
 *              TRANSA = 'N' or 'n'   op( A ) = A.
 *
 *              TRANSA = 'T' or 't'   op( A ) = A'.
 *
 *              TRANSA = 'C' or 'c'   op( A ) = A'.
 *
 *           Unchanged on exit.
 *
 *  DIAG   - CHARACTER*1.
 *           On entry, DIAG specifies whether or not A is unit triangular
 *           as follows:
 *
 *              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *
 *              DIAG = 'N' or 'n'   A is not assumed to be unit
 *                                  triangular.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry, M specifies the number of rows of B. M must be at
 *           least zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the number of columns of B.  N must be
 *           at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry,  ALPHA specifies the scalar  alpha. When  alpha is
 *           zero then  A is not referenced and  B need not be set before
 *           entry.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, k ), where k is m
 *           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
 *           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
 *           upper triangular part of the array  A must contain the upper
 *           triangular matrix  and the strictly lower triangular part of
 *           A is not referenced.
 *           Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
 *           lower triangular part of the array  A must contain the lower
 *           triangular matrix  and the strictly upper triangular part of
 *           A is not referenced.
 *           Note that when  DIAG = 'U' or 'u',  the diagonal elements of
 *           A  are not referenced either,  but are assumed to be  unity.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
 *           LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
 *           then LDA must be at least max( 1, n ).
 *           Unchanged on exit.
 *
 *  B      - DOUBLE PRECISION array of DIMENSION ( LDB, n ).
 *           Before entry,  the leading  m by n part of the array  B must
 *           contain  the  right-hand  side  matrix  B,  and  on exit  is
 *           overwritten by the solution matrix  X.
 *
 *  LDB    - INTEGER.
 *           On entry, LDB specifies the first dimension of B as declared
 *           in  the  calling  (sub)  program.   LDB  must  be  at  least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *
 *  Level 3 Blas routine.
 *
 *
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 *
 *
 *     .. External Functions ..
 **/
void C_DTRSM(char side, char uplo, char transa, char diag, int m, int n,
             double alpha, double *a, int lda, double *b, int ldb)
{
    if (m == 0 || n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DTRSM uplo argument is invalid.");
    if (side == 'L' || side == 'L')
        side = 'R';
    else if (side == 'R' || side == 'r')
        side = 'L';
    else
        throw std::invalid_argument("C_DTRSM side argument is invalid.");
    ::F_DTRSM(&side, &uplo, &transa, &diag, &n, &m, &alpha, a, &lda, b, &ldb);
}

/**
 *  Purpose
 *  =======
 *
 *  DTRSV  solves one of the systems of equations
 *
 *     A*x = b,   or   A'*x = b,
 *
 *  where b and x are n element vectors and A is an n by n unit, or
 *  non-unit, upper or lower triangular matrix.
 *
 *  No test for singularity or near-singularity is included in this
 *  routine. Such tests must be performed before calling this routine.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the matrix is an upper or
 *           lower triangular matrix as follows:
 *
 *              UPLO = 'U' or 'u'   A is an upper triangular matrix.
 *
 *              UPLO = 'L' or 'l'   A is a lower triangular matrix.
 *
 *           Unchanged on exit.
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the equations to be solved as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   A*x = b.
 *
 *              TRANS = 'T' or 't'   A'*x = b.
 *
 *              TRANS = 'C' or 'c'   A'*x = b.
 *
 *           Unchanged on exit.
 *
 *  DIAG   - CHARACTER*1.
 *           On entry, DIAG specifies whether or not A is unit
 *           triangular as follows:
 *
 *              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *
 *              DIAG = 'N' or 'n'   A is not assumed to be unit
 *                                  triangular.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
 *           upper triangular part of the array A must contain the upper
 *           triangular matrix and the strictly lower triangular part of
 *           A is not referenced.
 *           Before entry with UPLO = 'L' or 'l', the leading n by n
 *           lower triangular part of the array A must contain the lower
 *           triangular matrix and the strictly upper triangular part of
 *           A is not referenced.
 *           Note that when  DIAG = 'U' or 'u', the diagonal elements of
 *           A are not referenced either, but are assumed to be unity.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element right-hand side vector b. On exit, X is overwritten
 *           with the solution vector x.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *
 *  Level 2 Blas routine.
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *
 *     .. Parameters ..
 **/
void C_DTRSV(char uplo, char trans, char diag, int n, double *a, int lda,
             double *x, int incx)
{
    if (n == 0)
        return;
    if (uplo == 'U' || uplo == 'u')
        uplo = 'L';
    else if (uplo == 'L' || uplo == 'l')
        uplo = 'U';
    else
        throw std::invalid_argument("C_DTRSV uplo argument is invalid.");
    if (trans == 'N' || trans == 'n')
        trans = 'T';
    else if (trans == 'T' || trans == 't')
        trans = 'N';
    else
        throw std::invalid_argument("C_DTRSV trans argument is invalid.");
    ::F_DTRSV(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
}
}
