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
 * You should have received a copy of the GNU Lesser General Public License
 * along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#ifdef HAVE_TBLIS
#include <tblis/tblis.h>
#endif

#include "core.h"
#include "math/math.h"
#include "tensor/indices.h"
#include <algorithm>
#include <ambit/print.h>
#include <ambit/timer.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string.h>

namespace ambit
{

CoreTensorImpl::CoreTensorImpl(const string &name, const Dimension &dims)
    : TensorImpl(CoreTensor, name, dims)
{
    data_.resize(numel(), 0L);
}

void CoreTensorImpl::reshape(const Dimension &dims)
{
    TensorImpl::reshape(dims);
}

double CoreTensorImpl::norm(int type) const
{
    double val = 0.0;
    switch (type)
    {
    case 0:
        for (size_t ind = 0L; ind < numel(); ind++)
        {
            val = std::max(val, fabs(data_[ind]));
        }
        return val;
    case 1:
        for (size_t ind = 0L; ind < numel(); ind++)
        {
            val += fabs(data_[ind]);
        }
        return val;
    case 2:
        for (size_t ind = 0L; ind < numel(); ind++)
        {
            val += data_[ind] * data_[ind];
        }
        return sqrt(val);
    default:
        throw std::runtime_error(
            "Norm must be 0 (infty-norm), 1 (1-norm), or 2 (2-norm)");
        break;
    }
}

tuple<double, vector<size_t>> CoreTensorImpl::max() const
{
    tuple<double, vector<size_t>> element;

    std::get<0>(element) = std::numeric_limits<double>::lowest();

    citerate([&](const vector<size_t> &indices, const double &value) {
        if (std::get<0>(element) < value)
        {
            std::get<0>(element) = value;
            std::get<1>(element) = indices;
        }
    });

    return element;
}

tuple<double, vector<size_t>> CoreTensorImpl::min() const
{
    tuple<double, vector<size_t>> element;

    std::get<0>(element) = std::numeric_limits<double>::max();

    citerate([&](const vector<size_t> &indices, const double &value) {
        if (std::get<0>(element) > value)
        {
            std::get<0>(element) = value;
            std::get<1>(element) = indices;
        }
    });

    return element;
}

void CoreTensorImpl::scale(double beta)
{
    if (beta == 0.0)
        memset(data_.data(), '\0', sizeof(double) * numel());
    else
        C_DSCAL(numel(), beta, data_.data(), 1);
}

void CoreTensorImpl::set(double alpha)
{
    for (size_t i = 0; i < numel(); ++i)
        data_[i] = alpha;
}

namespace
{

std::string describe_tensor(ConstTensorImplPtr A, const Indices &Ainds)
{
    std::ostringstream buffer;
    buffer << A->name() << "[" << indices::to_string(Ainds) << "]";
    return buffer.str();
}
std::string describe_contraction(ConstTensorImplPtr C, const Indices &Cinds,
                                 ConstTensorImplPtr A, const Indices &Ainds,
                                 ConstTensorImplPtr B, const Indices &Binds,
                                 const double &alpha, const double &beta)
{
    std::ostringstream buffer;
    buffer << beta << " " << describe_tensor(C, Cinds) << " += ";
    buffer << alpha << " " << describe_tensor(A, Ainds) << " * "
           << describe_tensor(B, Binds);
    return buffer.str();
}

#ifdef HAVE_TBLIS
vector<tblis::label_type> labels_to_num_indices(const Indices &inds,
                                                vector<string> &labels)
{
    size_t n = inds.size();
    vector<tblis::label_type> num_inds(n);
    for (size_t k = 0; k < n; k++)
    {
        auto it = std::find(labels.begin(), labels.end(), inds[k]);
        if (it != labels.end())
        {
            num_inds[k] = std::distance(labels.begin(), it);
        }
        else
        {
            labels.push_back(inds[k]);
            num_inds[k] = labels.size() - 1;
        }
    }
    return num_inds;
}

vector<size_t> make_dims_vec(const vector<tblis::label_type> &labels,
                             const Dimension &dims,
                             const vector<tblis::label_type> &unique_labels)
{
    vector<size_t> dims_vec(unique_labels.size(), 0);
    size_t n = labels.size();
    for (size_t k = 0; k < n; k++)
    {
        auto it =
            std::find(unique_labels.begin(), unique_labels.end(), labels[k]);
        if (it != unique_labels.end())
        {
            dims_vec[std::distance(unique_labels.begin(), it)] = dims[k];
        }
    }
    return dims_vec;
}
#endif

} // anonymous namespace

void CoreTensorImpl::contract(ConstTensorImplPtr A, ConstTensorImplPtr B,
                              const Indices &Cinds, const Indices &Ainds,
                              const Indices &Binds, double alpha, double beta)
{
#define TBLIS_DEBUG 0
#ifdef HAVE_TBLIS
    ambit::timer::timer_push("pre-TBLIS: internal overhead");

    size_t Asize = Ainds.size();
    size_t Bsize = Binds.size();
    size_t Csize = Cinds.size();

    vector<string> labels;
    vector<tblis::label_type> A_labels = labels_to_num_indices(Ainds, labels);
    vector<tblis::label_type> B_labels = labels_to_num_indices(Binds, labels);
    vector<tblis::label_type> C_labels = labels_to_num_indices(Cinds, labels);

    // Determine unique indices
    vector<tblis::label_type> unique_labels;
    unique_labels.insert(unique_labels.end(), A_labels.begin(), A_labels.end());
    unique_labels.insert(unique_labels.end(), B_labels.begin(), B_labels.end());
    unique_labels.insert(unique_labels.end(), C_labels.begin(), C_labels.end());
    std::sort(unique_labels.begin(), unique_labels.end());
    auto it = std::unique(unique_labels.begin(), unique_labels.end());
    unique_labels.resize(std::distance(unique_labels.begin(), it));

    vector<size_t> A_dims_vec =
        make_dims_vec(A_labels, A->dims(), unique_labels);
    vector<size_t> B_dims_vec =
        make_dims_vec(B_labels, B->dims(), unique_labels);
    vector<size_t> C_dims_vec =
        make_dims_vec(C_labels, this->dims(), unique_labels);

    for (size_t ind = 0; ind < unique_labels.size(); ind++)
    {
//        std::cout << "\n " << (char)(unique_labels[ind] + 65);
//        std::cout << "  " << A_dims_vec[ind];
//        std::cout << "  " << B_dims_vec[ind];
//        std::cout << "  " << C_dims_vec[ind];
//    std::cout << std::endl;
        auto A_dim = A_dims_vec[ind];
        auto B_dim = B_dims_vec[ind];
        auto C_dim = C_dims_vec[ind];
        if (((A_dim == 0) and (B_dim == 0)) or
            ((A_dim == 0) and (C_dim == 0)) or ((B_dim == 0) and (C_dim == 0)))
        {
            throw std::runtime_error(
                "Invalid contraction topology - index only occurs "
                "once.\n" +
                describe_contraction(this, Cinds, A, Ainds, B, Binds, alpha,
                                     beta));
        }
        if ((B_dim == 0) and (A_dim != C_dim))
        {
            throw std::runtime_error("Invalid AC (Left) index size\n" +
                                     describe_contraction(this, Cinds, A, Ainds,
                                                          B, Binds, alpha,
                                                          beta));
        }
        if ((A_dim == 0) and (B_dim != C_dim))
        {
            throw std::runtime_error("Invalid BC (Right) index size\n" +
                                     describe_contraction(this, Cinds, A, Ainds,
                                                          B, Binds, alpha,
                                                          beta));
        }
        if ((C_dim == 0) and (A_dim != B_dim))
        {
            throw std::runtime_error("Invalid AB (Contraction) index size\n" +
                                     describe_contraction(this, Cinds, A, Ainds,
                                                          B, Binds, alpha,
                                                          beta));
        }
        if ((A_dim * B_dim * C_dim != 0) and
            ((A_dim != B_dim) or (B_dim != C_dim)))
        {
            throw std::runtime_error("Invalid ABC (Hadamard) index size\n" +
                                     describe_contraction(this, Cinds, A, Ainds, B,
                                                          Binds, alpha, beta));
        }
    }

    std::string indices_A = indices::to_string(Ainds, "");
    std::string indices_B = indices::to_string(Binds, "");
    std::string indices_C = indices::to_string(Cinds, "");

#if TBLIS_DEBUG
    std::cout << "Calling TBLIS" << std::endl;
    std::cout << "\n indices_A " << indices_A;
    auto Adims = A->dims();
    for (auto d : Adims)
        std::cout << " " << d;
    std::cout << std::endl;
    std::cout << "\n indices_B " << indices_B;
    auto Bdims = B->dims();
    for (auto d : Bdims)
        std::cout << " " << d;
    std::cout << std::endl;
    std::cout << "\n indices_C " << indices_C;
    auto Cdims = this->dims();
    for (auto d : Cdims)
        std::cout << " " << d;
    std::cout << std::endl;
#endif

ambit::timer::timer_pop();

    ambit::timer::timer_push("BLAS");

    // Scalar multiplication
    if ((Asize == 0) and (Bsize == 0) and (Csize == 0))
    {
        data_[0] = alpha * A->data()[0] * B->data()[0] + beta * data_[0];
    }
    // Index Type A -> add operation
    else if ((Asize == Csize) and (Bsize == 0))
    {
        TensorImplPtr Ap = const_cast<TensorImplPtr>(A);
        MArray::varray_view<double> A_v(Ap->dims(), &(Ap->data()[0]));
        MArray::varray_view<double> C_v(this->dims(), &(this->data()[0]));
        tblis::add<double>(alpha * B->data()[0], A_v, indices_A.c_str(), beta,
                           C_v, indices_C.c_str());
    }
    // Index Type B -> add operation
    else if ((Asize == 0) and (Bsize == Csize))
    {
        TensorImplPtr Bp = const_cast<TensorImplPtr>(B);
        MArray::varray_view<double> B_v(Bp->dims(), &(Bp->data()[0]));
        MArray::varray_view<double> C_v(this->dims(), &(this->data()[0]));
        tblis::add<double>(alpha * A->data()[0], B_v, indices_B.c_str(), beta,
                           C_v, indices_C.c_str());
    }
    // Index Type AB -> dot operation
    else if ((Asize > 0) and (Bsize > 0) and (Csize == 0))
    {
        TensorImplPtr Ap = const_cast<TensorImplPtr>(A);
        TensorImplPtr Bp = const_cast<TensorImplPtr>(B);
        MArray::varray_view<double> A_v(Ap->dims(), &(Ap->data()[0]));
        MArray::varray_view<double> B_v(Bp->dims(), &(Bp->data()[0]));
        data_[0] = alpha * tblis::dot<double>(A_v, indices_A.c_str(), B_v,
                                              indices_B.c_str()) +
                   beta * data_[0];
    }
    // Index Type ABC -> multp operation
    else if ((Asize > 0) and (Bsize > 0) and (Csize > 0))
    {
#if TBLIS_DEBUG
        std::cout << "\n Calling mult " << std::endl;
        std::cout << "\n Get Ap " << std::endl;
        std::cout << "\n Get Bp " << std::endl;
        std::cout << "\n Get A_v " << std::endl;
        std::cout << "\n Get B_v " << std::endl;
        std::cout << "\n Get C_v " << std::endl;
        std::cout << "\n Call mult " << std::endl;
#endif
        TensorImplPtr Ap = const_cast<TensorImplPtr>(A);
        TensorImplPtr Bp = const_cast<TensorImplPtr>(B);
        MArray::varray_view<double> A_v(Ap->dims(), &(Ap->data()[0]));
        MArray::varray_view<double> B_v(Bp->dims(), &(Bp->data()[0]));
        MArray::varray_view<double> C_v(this->dims(), &(this->data()[0]));
        tblis::mult<double>(alpha, A_v, &(A_labels.data()[0]), B_v,
                            &(B_labels.data()[0]), beta, C_v,
                            &(C_labels.data()[0]));
    }
    ambit::timer::timer_pop();
#else
    shared_ptr<TensorImpl> A2;
    shared_ptr<TensorImpl> B2;
    shared_ptr<TensorImpl> C2;
    contract(A, B, Cinds, Ainds, Binds, A2, B2, C2, alpha, beta);
#endif
} // namespace ambit

void CoreTensorImpl::contract(ConstTensorImplPtr A, ConstTensorImplPtr B,
                              const Indices &Cinds, const Indices &Ainds,
                              const Indices &Binds,
                              std::shared_ptr<TensorImpl> &A2,
                              std::shared_ptr<TensorImpl> &B2,
                              std::shared_ptr<TensorImpl> &C2, double alpha,
                              double beta)
{
    ambit::timer::timer_push("pre-BLAS: internal overhead");

    TensorImplPtr C = this;

    // => Permutation Logic <= //

    // Determine unique indices
    Indices inds;
    inds.insert(inds.end(), Cinds.begin(), Cinds.end());
    inds.insert(inds.end(), Ainds.begin(), Ainds.end());
    inds.insert(inds.end(), Binds.begin(), Binds.end());
    std::sort(inds.begin(), inds.end());
    Indices::iterator it = std::unique(inds.begin(), inds.end());
    inds.resize(std::distance(inds.begin(), it));

    // Determine index types and positions (and GEMM sizes while we are there)
    size_t ABC_size = 1L;
    size_t BC_size = 1L;
    size_t AC_size = 1L;
    size_t AB_size = 1L;
    Indices compound_names = {"PC", "PA", "PB", "iC", "iA",
                              "jC", "jB", "kA", "kB"};
    map<string, vector<pair<int, string>>> compound_inds;
    for (size_t ind = 0L; ind < compound_names.size(); ind++)
    {
        compound_inds[compound_names[ind]] = vector<pair<int, string>>();
    }
    for (size_t ind = 0L; ind < inds.size(); ind++)
    {
        string index = inds[ind];
        int Cpos = indices::find_index_in_vector(Cinds, index);
        int Apos = indices::find_index_in_vector(Ainds, index);
        int Bpos = indices::find_index_in_vector(Binds, index);

        if (Cpos != -1)
        {
            if (Apos != -1)
            {
                if (Bpos != -1)
                {
                    if (C->dims()[Cpos] != A->dims()[Apos] ||
                        C->dims()[Cpos] != B->dims()[Bpos])
                        throw std::runtime_error(
                            "Invalid ABC (Hadamard) index size\n" +
                            describe_contraction(C, Cinds, A, Ainds, B, Binds,
                                                 alpha, beta));
                    compound_inds["PC"].push_back(std::make_pair(Cpos, index));
                    compound_inds["PA"].push_back(std::make_pair(Apos, index));
                    compound_inds["PB"].push_back(std::make_pair(Bpos, index));
                    ABC_size *= C->dims()[Cpos];
                }
                else
                {
                    if (C->dims()[Cpos] != A->dims()[Apos])
                        throw std::runtime_error(
                            "Invalid AC (Left) index size\n" +
                            describe_contraction(C, Cinds, A, Ainds, B, Binds,
                                                 alpha, beta));
                    compound_inds["iC"].push_back(std::make_pair(Cpos, index));
                    compound_inds["iA"].push_back(std::make_pair(Apos, index));
                    AC_size *= C->dims()[Cpos];
                }
            }
            else
            {
                if (Bpos != -1)
                {
                    if (C->dims()[Cpos] != B->dims()[Bpos])
                        throw std::runtime_error(
                            "Invalid BC (Right) index size\n" +
                            describe_contraction(C, Cinds, A, Ainds, B, Binds,
                                                 alpha, beta));
                    compound_inds["jC"].push_back(std::make_pair(Cpos, index));
                    compound_inds["jB"].push_back(std::make_pair(Bpos, index));
                    BC_size *= C->dims()[Cpos];
                }
                else
                {
                    throw std::runtime_error(
                        "Invalid contraction topology - index only occurs "
                        "once.\n" +
                        describe_contraction(C, Cinds, A, Ainds, B, Binds,
                                             alpha, beta));
                }
            }
        }
        else
        {
            if (Apos != -1 && Bpos != -1)
            {
                if (A->dims()[Apos] != B->dims()[Bpos])
                    throw std::runtime_error(
                        "Invalid AB (Contraction) index size\n" +
                        describe_contraction(C, Cinds, A, Ainds, B, Binds,
                                             alpha, beta));
                compound_inds["kA"].push_back(std::make_pair(Apos, index));
                compound_inds["kB"].push_back(std::make_pair(Bpos, index));
                AB_size *= B->dims()[Bpos];
            }
            else
            {
                throw std::runtime_error(
                    "Invalid contraction topology - index only occurs once.\n" +
                    describe_contraction(C, Cinds, A, Ainds, B, Binds, alpha,
                                         beta));
            }
        }
    }

    // Sort compound indices by primitive indices to determine continuity (will
    // be sequential if continuous)
    for (size_t ind = 0L; ind < compound_names.size(); ind++)
    {
        std::sort(compound_inds[compound_names[ind]].begin(),
                  compound_inds[compound_names[ind]].end());
    }

    // Flags to mark if tensors must be permuted
    // Contiguous Index Test (always requires permutation)
    bool permC = !indices::contiguous(compound_inds["PC"]);
    permC = permC || !indices::contiguous(compound_inds["iC"]);
    permC = permC || !indices::contiguous(compound_inds["jC"]);
    bool permA = !indices::contiguous(compound_inds["PA"]);
    permA = permA || !indices::contiguous(compound_inds["iA"]);
    permA = permA || !indices::contiguous(compound_inds["kA"]);
    bool permB = !indices::contiguous(compound_inds["PB"]);
    permB = permB || !indices::contiguous(compound_inds["jB"]);
    permB = permB || !indices::contiguous(compound_inds["kB"]);

    // Hadamard Test (always requires permutation)
    size_t Psize = compound_inds["PC"].size();
    if (Psize)
    {
        permC = permC || (compound_inds["PC"][0].first != 0);
        permA = permA || (compound_inds["PA"][0].first != 0);
        permB = permB || (compound_inds["PB"][0].first != 0);
    }

    /// Figure out the initial transposes (will be fixed if perm is set)
    bool A_transpose =
        compound_inds["iA"].size() && compound_inds["iA"][0].first != Psize;
    bool B_transpose =
        compound_inds["jB"].size() && compound_inds["jB"][0].first == Psize;
    bool C_transpose =
        compound_inds["iC"].size() && compound_inds["iC"][0].first != Psize;

    // Fix contiguous considerations (already in correct order for contiguous
    // cases)
    map<string, vector<string>> compound_inds2;
    for (size_t ind = 0L; ind < compound_names.size(); ind++)
    {
        string key = compound_names[ind];
        vector<string> vals;
        for (size_t ind2 = 0L; ind2 < compound_inds[key].size(); ind2++)
        {
            vals.push_back(compound_inds[key][ind2].second);
        }
        compound_inds2[key] = vals;
    }

    /**
     * Fix permutation order considerations
     *
     * Rules if a permutation mismatch is detected:
     * -If both tensors are already on the permute list, it doesn't matter which
     *is fixed
     * -Else if one tensor is already on the permute list but not the other, fix
     *the one that is already on the permute list
     * -Else fix the smaller tensor
     *
     * Note: this scheme is not optimal is permutation mismatches exist in P -
     *for reasons of simplicity, A and B are
     * permuted to C's P order, with no present considerations of better
     *pathways
     **/
    if (!indices::equivalent(compound_inds2["iC"], compound_inds2["iA"]))
    {
        if (permC)
        {
            compound_inds2["iC"] = compound_inds2["iA"];
        }
        else if (permA)
        {
            compound_inds2["iA"] = compound_inds2["iC"];
        }
        else if (C->numel() <= A->numel())
        {
            compound_inds2["iC"] = compound_inds2["iA"];
            permC = true;
        }
        else
        {
            compound_inds2["iA"] = compound_inds2["iC"];
            permA = true;
        }
    }
    if (!indices::equivalent(compound_inds2["jC"], compound_inds2["jB"]))
    {
        if (permC)
        {
            compound_inds2["jC"] = compound_inds2["jB"];
        }
        else if (permB)
        {
            compound_inds2["jB"] = compound_inds2["jC"];
        }
        else if (C->numel() <= B->numel())
        {
            compound_inds2["jC"] = compound_inds2["jB"];
            permC = true;
        }
        else
        {
            compound_inds2["jB"] = compound_inds2["jC"];
            permB = true;
        }
    }
    if (!indices::equivalent(compound_inds2["kA"], compound_inds2["kB"]))
    {
        if (permA)
        {
            compound_inds2["kA"] = compound_inds2["kB"];
        }
        else if (permB)
        {
            compound_inds2["kB"] = compound_inds2["kA"];
        }
        else if (A->numel() <= B->numel())
        {
            compound_inds2["kA"] = compound_inds2["kB"];
            permA = true;
        }
        else
        {
            compound_inds2["kB"] = compound_inds2["kA"];
            permB = true;
        }
    }
    if (!indices::equivalent(compound_inds2["PC"], compound_inds2["PA"]))
    {
        compound_inds2["PA"] = compound_inds2["PC"];
        permA = true;
    }
    if (!indices::equivalent(compound_inds2["PC"], compound_inds2["PB"]))
    {
        compound_inds2["PB"] = compound_inds2["PC"];
        permB = true;
    }

    /// Assign the permuted indices (if flagged for permute) or the original
    /// indices
    Indices Cinds2;
    Indices Ainds2;
    Indices Binds2;
    if (permC)
    {
        Cinds2.insert(Cinds2.end(), compound_inds2["PC"].begin(),
                      compound_inds2["PC"].end());
        Cinds2.insert(Cinds2.end(), compound_inds2["iC"].begin(),
                      compound_inds2["iC"].end());
        Cinds2.insert(Cinds2.end(), compound_inds2["jC"].begin(),
                      compound_inds2["jC"].end());
        C_transpose = false;
    }
    else
    {
        Cinds2 = Cinds;
    }
    if (permA)
    {
        Ainds2.insert(Ainds2.end(), compound_inds2["PA"].begin(),
                      compound_inds2["PA"].end());
        Ainds2.insert(Ainds2.end(), compound_inds2["iA"].begin(),
                      compound_inds2["iA"].end());
        Ainds2.insert(Ainds2.end(), compound_inds2["kA"].begin(),
                      compound_inds2["kA"].end());
        A_transpose = false;
    }
    else
    {
        Ainds2 = Ainds;
    }
    if (permB)
    {
        Binds2.insert(Binds2.end(), compound_inds2["PB"].begin(),
                      compound_inds2["PB"].end());
        Binds2.insert(Binds2.end(), compound_inds2["jB"].begin(),
                      compound_inds2["jB"].end());
        Binds2.insert(Binds2.end(), compound_inds2["kB"].begin(),
                      compound_inds2["kB"].end());
        B_transpose = true;
    }
    else
    {
        Binds2 = Binds;
    }

    // So what exactly happened?
    /**
    printf("==> Core Contraction <==\n\n");
    printf("Original: C[");
    for (size_t ind = 0l; ind < Cinds.size(); ind++) {
        printf("%s", Cinds[ind].c_str());
    }
    printf("] = A[");
    for (size_t ind = 0l; ind < Ainds.size(); ind++) {
        printf("%s", Ainds[ind].c_str());
    }
    printf("] * B[");
    for (size_t ind = 0l; ind < Binds.size(); ind++) {
        printf("%s", Binds[ind].c_str());
    }
    printf("]\n");
    printf("New:      C[");
    for (size_t ind = 0l; ind < Cinds2.size(); ind++) {
        printf("%s", Cinds2[ind].c_str());
    }
    printf("] = A[");
    for (size_t ind = 0l; ind < Ainds2.size(); ind++) {
        printf("%s", Ainds2[ind].c_str());
    }
    printf("] * B[");
    for (size_t ind = 0l; ind < Binds2.size(); ind++) {
        printf("%s", Binds2[ind].c_str());
    }
    printf("]\n");
    printf("C Permuted: %s\n", permC ? "Yes" : "No");
    printf("A Permuted: %s\n", permA ? "Yes" : "No");
    printf("B Permuted: %s\n", permB ? "Yes" : "No");
    printf("\n");
    **/

    ambit::timer::timer_pop();

    // => Alias or Allocate A, B, C <= //
    // => Permute A, B, and C if Necessary <= //

    double *Cp = ((CoreTensorImplPtr)C)->data().data();
    double *Ap = ((CoreTensorImplPtr)A)->data().data();
    double *Bp = ((CoreTensorImplPtr)B)->data().data();
    double *C2p = Cp;
    double *A2p = Ap;
    double *B2p = Bp;

    if (permC)
    {
        ambit::timer::timer_push("pre-BLAS: internal C allocation");
        if (!C2)
        {
            Dimension Cdims2 =
                indices::permuted_dimension(C->dims(), Cinds2, Cinds);
            C2 = std::make_shared<CoreTensorImpl>("C2", Cdims2);
        }
        C2p = C2->data().data();
        ambit::timer::timer_pop();
        if (beta != 0.0)
        {
            ambit::timer::timer_push("pre-BLAS: internal C permutation");
            C2->permute(C, Cinds2, Cinds);
            ambit::timer::timer_pop();
        }
    }
    if (permA)
    {
        ambit::timer::timer_push("pre-BLAS: internal A allocation");
        if (!A2)
        {
            Dimension Adims2 =
                indices::permuted_dimension(A->dims(), Ainds2, Ainds);
            A2 = std::make_shared<CoreTensorImpl>("A2", Adims2);
        }
        A2p = A2->data().data();
        ambit::timer::timer_pop();
        ambit::timer::timer_push("pre-BLAS: internal A permutation");
        A2->permute(A, Ainds2, Ainds);
        ambit::timer::timer_pop();
    }
    if (permB)
    {
        ambit::timer::timer_push("pre-BLAS: internal B allocation");
        if (!B2)
        {
            Dimension Bdims2 =
                indices::permuted_dimension(B->dims(), Binds2, Binds);
            B2 = std::make_shared<CoreTensorImpl>("B2", Bdims2);
        }
        B2p = B2->data().data();
        ambit::timer::timer_pop();
        ambit::timer::timer_push("pre-BLAS: internal B permutation");
        B2->permute(B, Binds2, Binds);
        ambit::timer::timer_pop();
    }

    // => GEMM Indexing <= //

    // => GEMM <= //

    ambit::timer::timer_push("BLAS");
    for (size_t P = 0L; P < ABC_size; P++)
    {

        char transL;
        char transR;
        size_t nrow;
        size_t ncol;
        double *Lp;
        double *Rp;
        size_t ldaL;
        size_t ldaR;

        if (C_transpose)
        {
            Lp = B2p;
            Rp = A2p;
            nrow = BC_size;
            ncol = AC_size;
            transL = (B_transpose ? 'N' : 'T');
            transR = (A_transpose ? 'N' : 'T');
            ldaL = (B_transpose ? AB_size : BC_size);
            ldaR = (A_transpose ? AC_size : AB_size);
        }
        else
        {
            Lp = A2p;
            Rp = B2p;
            nrow = AC_size;
            ncol = BC_size;
            transL = (A_transpose ? 'T' : 'N');
            transR = (B_transpose ? 'T' : 'N');
            ldaL = (A_transpose ? AC_size : AB_size);
            ldaR = (B_transpose ? AB_size : BC_size);
        }

        size_t nzip = AB_size;
        size_t ldaC = (C_transpose ? AC_size : BC_size);

        if (nrow != 1L)
        {
            if (ncol != 1L)
            {
                if (nzip != 1L)
                {
                    C_DGEMM(transL, transR, nrow, ncol, nzip, alpha, Lp, ldaL,
                            Rp, ldaR, beta, C2p, ldaC);
                }
                else
                {
                    for (size_t row = 0L; row < nrow; row++)
                    {
                        C_DSCAL(ncol, beta, C2p + row * ldaC, 1);
                    }
                    C_DGER(nrow, ncol, alpha, Lp, 1, Rp, 1, C2p, ldaC);
                }
            }
            else
            {
                if (nzip != 1L)
                {
                    if (transL == 'N')
                    {
                        C_DGEMV('N', nrow, nzip, alpha, Lp, ldaL, Rp, 1, beta,
                                C2p, 1);
                    }
                    else
                    {
                        C_DGEMV('T', nzip, nrow, alpha, Lp, ldaL, Rp, 1, beta,
                                C2p, 1);
                    }
                }
                else
                {
                    C_DSCAL(nrow, beta, C2p, 1);
                    C_DAXPY(nrow, alpha * (*Rp), Lp, 1, C2p, 1);
                }
            }
        }
        else
        {
            if (ncol != 1L)
            {
                if (nzip != 1L)
                {
                    if (transR == 'N')
                    {
                        C_DGEMV('T', nzip, ncol, alpha, Rp, ldaR, Lp, 1, beta,
                                C2p, 1);
                    }
                    else
                    {
                        C_DGEMV('N', ncol, nzip, alpha, Rp, ldaR, Lp, 1, beta,
                                C2p, 1);
                    }
                }
                else
                {
                    C_DSCAL(ncol, beta, C2p, 1);
                    C_DAXPY(ncol, alpha * (*Lp), Rp, 1, C2p, 1);
                }
            }
            else
            {
                if (nzip != 1L)
                {
                    (*C2p) *= beta;
                    (*C2p) += alpha * C_DDOT(nzip, Lp, 1, Rp, 1);
                }
                else
                {
                    (*C2p) = alpha * (*Lp) * (*Rp) + beta * (*C2p);
                }
            }
        }

        C2p += AC_size * BC_size;
        A2p += AB_size * AC_size;
        B2p += AB_size * BC_size;
    }
    ambit::timer::timer_pop();

    // => Permute C if Necessary <= //

    if (permC)
    {
        ambit::timer::timer_push("post-BLAS: internal C permutation");
        C->permute(C2.get(), Cinds, Cinds2);
        ambit::timer::timer_pop();
    }
}

void CoreTensorImpl::permute(ConstTensorImplPtr A, const Indices &CindsS,
                             const Indices &AindsS, double alpha, double beta)
{
    ambit::timer::timer_push("P: " + std::to_string(beta) + " " + A->name() +
                             "[" + indices::to_string(CindsS) +
                             "] = " + std::to_string(alpha) + " " + A->name() +
                             "[" + indices::to_string(AindsS) + "]");

    // => Convert to indices of A <= //

    vector<size_t> Ainds = indices::permutation_order(CindsS, AindsS);
    for (size_t dim = 0; dim < rank(); dim++)
    {
        if (dims()[dim] != A->dims()[Ainds[dim]])
            throw std::runtime_error(
                "Permuted tensors do not have same dimensions");
    }

    /// Data pointers
    double *Cp = data().data();
    double *Ap = ((const CoreTensorImplPtr)A)->data().data();

    /// Beta scale
    scale(beta);

    // => Index Logic <= //

    /// Determine the number of united fast indices and memcpy size
    /// C_ij = A_ji would have no fast dimensions and a fast size of 1
    /// C_ijk = A_jik would have k as a fast dimension, and a fast size of
    /// dim(k)
    /// C_ijkl = A_jikl would have k and l as fast dimensions, and a fast size
    /// of dim(k) * dim(l)
    int fast_dims = 0;
    size_t fast_size = 1L;
    for (int dim = ((int)rank()) - 1; dim >= 0; dim--)
    {
        if (dim == ((int)Ainds[dim]))
        {
            fast_dims++;
            fast_size *= dims()[dim];
        }
        else
        {
            break;
        }
    }

    /// Determine the total number of memcpy operations
    int slow_dims = rank() - fast_dims;

    /// Fully sorted case or (equivalently) rank-0 or rank-1 tensors
    if (slow_dims == 0)
    {
        //::memcpy(Cp,Ap,sizeof(double)*fast_size);
        C_DAXPY(fast_size, alpha, Ap, 1, Cp, 1);
        ambit::timer::timer_pop();
        return;
    }

    /// Number of collapsed indices in permutation traverse
    size_t slow_size = 1L;
    for (int dim = 0; dim < slow_dims; dim++)
    {
        slow_size *= dims()[dim];
    }

    /// Strides of slow indices of tensor A in its own ordering
    vector<size_t> Astrides(slow_dims, 0L);
    if (slow_dims != 0)
        Astrides[slow_dims - 1] = fast_size;
    for (int dim = slow_dims - 2; dim >= 0; dim--)
    {
        Astrides[dim] = Astrides[dim + 1] * A->dims()[dim + 1];
    }

    /// Strides of slow indices of tensor A in the ordering of tensor C
    vector<size_t> AstridesC(slow_dims, 0L);
    for (int dim = 0; dim < slow_dims; dim++)
    {
        AstridesC[dim] = Astrides[Ainds[dim]];
    }

    /// Strides of slow indices of tensor C in its own ordering
    vector<size_t> Cstrides(slow_dims, 0L);
    if (slow_dims != 0)
        Cstrides[slow_dims - 1] = fast_size;
    for (int dim = slow_dims - 2; dim >= 0; dim--)
    {
        Cstrides[dim] = Cstrides[dim + 1] * dims()[dim + 1];
    }

    /// Handle to dimensions of C
    const Dimension &Csizes = dims();

    // => Permute Operation <= //

    if (fast_size == 1L)
    {
        if (slow_dims == 2)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    double *Atp =
                        Ap + Cind0 * AstridesC[0] + Cind1 * AstridesC[1];
                    //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                    (*Ctp) += alpha * (*Atp);
                    Ctp += fast_size;
                }
            }
        }
        else if (slow_dims == 3)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        double *Atp = Ap + Cind0 * AstridesC[0] +
                                      Cind1 * AstridesC[1] +
                                      Cind2 * AstridesC[2];
                        //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                        (*Ctp) += alpha * (*Atp);
                        Ctp += fast_size;
                    }
                }
            }
        }
        else if (slow_dims == 4)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            double *Atp = Ap + Cind0 * AstridesC[0] +
                                          Cind1 * AstridesC[1] +
                                          Cind2 * AstridesC[2] +
                                          Cind3 * AstridesC[3];
                            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                            (*Ctp) += alpha * (*Atp);
                            Ctp += fast_size;
                        }
                    }
                }
            }
        }
        else if (slow_dims == 5)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++)
                            {
                                double *Atp = Ap + Cind0 * AstridesC[0] +
                                              Cind1 * AstridesC[1] +
                                              Cind2 * AstridesC[2] +
                                              Cind3 * AstridesC[3] +
                                              Cind4 * AstridesC[4];
                                //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                                (*Ctp) += alpha * (*Atp);
                                Ctp += fast_size;
                            }
                        }
                    }
                }
            }
        }
        else if (slow_dims == 6)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++)
                            {
                                for (size_t Cind5 = 0L; Cind5 < Csizes[5];
                                     Cind5++)
                                {
                                    double *Atp = Ap + Cind0 * AstridesC[0] +
                                                  Cind1 * AstridesC[1] +
                                                  Cind2 * AstridesC[2] +
                                                  Cind3 * AstridesC[3] +
                                                  Cind4 * AstridesC[4] +
                                                  Cind5 * AstridesC[5];
                                    //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                                    (*Ctp) += alpha * (*Atp);
                                    Ctp += fast_size;
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
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++)
                            {
                                for (size_t Cind5 = 0L; Cind5 < Csizes[5];
                                     Cind5++)
                                {
                                    for (size_t Cind6 = 0L; Cind6 < Csizes[6];
                                         Cind6++)
                                    {
                                        double *Atp = Ap +
                                                      Cind0 * AstridesC[0] +
                                                      Cind1 * AstridesC[1] +
                                                      Cind2 * AstridesC[2] +
                                                      Cind3 * AstridesC[3] +
                                                      Cind4 * AstridesC[4] +
                                                      Cind5 * AstridesC[5] +
                                                      Cind6 * AstridesC[6];
                                        //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                                        (*Ctp) += alpha * (*Atp);
                                        Ctp += fast_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (slow_dims == 8)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++)
                            {
                                for (size_t Cind5 = 0L; Cind5 < Csizes[5];
                                     Cind5++)
                                {
                                    for (size_t Cind6 = 0L; Cind6 < Csizes[6];
                                         Cind6++)
                                    {
                                        for (size_t Cind7 = 0L;
                                             Cind7 < Csizes[7]; Cind7++)
                                        {
                                            double *Atp = Ap +
                                                          Cind0 * AstridesC[0] +
                                                          Cind1 * AstridesC[1] +
                                                          Cind2 * AstridesC[2] +
                                                          Cind3 * AstridesC[3] +
                                                          Cind4 * AstridesC[4] +
                                                          Cind5 * AstridesC[5] +
                                                          Cind6 * AstridesC[6] +
                                                          Cind7 * AstridesC[7];
                                            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                                            (*Ctp) += alpha * (*Atp);
                                            Ctp += fast_size;
                                        }
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
                double *Ctp = Cp + ind * fast_size;
                double *Atp = Ap;
                size_t num = ind;
                for (int dim = slow_dims - 1; dim >= 0; dim--)
                {
                    size_t val = num % Csizes[dim]; // value of the dim-th index
                    num /= Csizes[dim];
                    Atp += val * AstridesC[dim];
                }
                //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                (*Ctp) += alpha * (*Atp);
            }
        }
    }
    else
    {
        if (slow_dims == 2)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    double *Atp =
                        Ap + Cind0 * AstridesC[0] + Cind1 * AstridesC[1];
                    //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                    C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                    Ctp += fast_size;
                }
            }
        }
        else if (slow_dims == 3)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        double *Atp = Ap + Cind0 * AstridesC[0] +
                                      Cind1 * AstridesC[1] +
                                      Cind2 * AstridesC[2];
                        //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                        C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                        Ctp += fast_size;
                    }
                }
            }
        }
        else if (slow_dims == 4)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            double *Atp = Ap + Cind0 * AstridesC[0] +
                                          Cind1 * AstridesC[1] +
                                          Cind2 * AstridesC[2] +
                                          Cind3 * AstridesC[3];
                            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                            C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                            Ctp += fast_size;
                        }
                    }
                }
            }
        }
        else if (slow_dims == 5)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++)
                            {
                                double *Atp = Ap + Cind0 * AstridesC[0] +
                                              Cind1 * AstridesC[1] +
                                              Cind2 * AstridesC[2] +
                                              Cind3 * AstridesC[3] +
                                              Cind4 * AstridesC[4];
                                //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                                C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                                Ctp += fast_size;
                            }
                        }
                    }
                }
            }
        }
        else if (slow_dims == 6)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++)
                            {
                                for (size_t Cind5 = 0L; Cind5 < Csizes[5];
                                     Cind5++)
                                {
                                    double *Atp = Ap + Cind0 * AstridesC[0] +
                                                  Cind1 * AstridesC[1] +
                                                  Cind2 * AstridesC[2] +
                                                  Cind3 * AstridesC[3] +
                                                  Cind4 * AstridesC[4] +
                                                  Cind5 * AstridesC[5];
                                    //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                                    C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
                                    Ctp += fast_size;
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
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++)
                            {
                                for (size_t Cind5 = 0L; Cind5 < Csizes[5];
                                     Cind5++)
                                {
                                    for (size_t Cind6 = 0L; Cind6 < Csizes[6];
                                         Cind6++)
                                    {
                                        double *Atp = Ap +
                                                      Cind0 * AstridesC[0] +
                                                      Cind1 * AstridesC[1] +
                                                      Cind2 * AstridesC[2] +
                                                      Cind3 * AstridesC[3] +
                                                      Cind4 * AstridesC[4] +
                                                      Cind5 * AstridesC[5] +
                                                      Cind6 * AstridesC[6];
                                        //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                                        C_DAXPY(fast_size, alpha, Atp, 1, Ctp,
                                                1);
                                        Ctp += fast_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (slow_dims == 8)
        {
#pragma omp parallel for
            for (size_t Cind0 = 0L; Cind0 < Csizes[0]; Cind0++)
            {
                double *Ctp = Cp + Cind0 * Cstrides[0];
                for (size_t Cind1 = 0L; Cind1 < Csizes[1]; Cind1++)
                {
                    for (size_t Cind2 = 0L; Cind2 < Csizes[2]; Cind2++)
                    {
                        for (size_t Cind3 = 0L; Cind3 < Csizes[3]; Cind3++)
                        {
                            for (size_t Cind4 = 0L; Cind4 < Csizes[4]; Cind4++)
                            {
                                for (size_t Cind5 = 0L; Cind5 < Csizes[5];
                                     Cind5++)
                                {
                                    for (size_t Cind6 = 0L; Cind6 < Csizes[6];
                                         Cind6++)
                                    {
                                        for (size_t Cind7 = 0L;
                                             Cind7 < Csizes[7]; Cind7++)
                                        {
                                            double *Atp = Ap +
                                                          Cind0 * AstridesC[0] +
                                                          Cind1 * AstridesC[1] +
                                                          Cind2 * AstridesC[2] +
                                                          Cind3 * AstridesC[3] +
                                                          Cind4 * AstridesC[4] +
                                                          Cind5 * AstridesC[5] +
                                                          Cind6 * AstridesC[6] +
                                                          Cind7 * AstridesC[7];
                                            //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                                            C_DAXPY(fast_size, alpha, Atp, 1,
                                                    Ctp, 1);
                                            Ctp += fast_size;
                                        }
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
                double *Ctp = Cp + ind * fast_size;
                double *Atp = Ap;
                size_t num = ind;
                for (int dim = slow_dims - 1; dim >= 0; dim--)
                {
                    size_t val = num % Csizes[dim]; // value of the dim-th index
                    num /= Csizes[dim];
                    Atp += val * AstridesC[dim];
                }
                //::memcpy(Ctp,Atp,sizeof(double)*fast_size);
                C_DAXPY(fast_size, alpha, Atp, 1, Ctp, 1);
            }
        }
    }

    ambit::timer::timer_pop();
}
void CoreTensorImpl::gemm(ConstTensorImplPtr A, ConstTensorImplPtr B,
                          bool transA, bool transB, size_t nrow, size_t ncol,
                          size_t nzip, size_t ldaA, size_t ldaB, size_t ldaC,
                          size_t offA, size_t offB, size_t offC, double alpha,
                          double beta)
{
    double *Cp = data().data();
    double *Ap = ((const CoreTensorImplPtr)A)->data().data();
    double *Bp = ((const CoreTensorImplPtr)B)->data().data();

    C_DGEMM((transA ? 'T' : 'N'), (transB ? 'T' : 'N'), nrow, ncol, nzip, alpha,
            Ap + offA, ldaA, Bp + offB, ldaB, beta, Cp + offC, ldaC);
}

map<string, TensorImplPtr> CoreTensorImpl::syev(EigenvalueOrder order) const
{
    squareCheck(this, true);

    CoreTensorImpl *vecs =
        new CoreTensorImpl("Eigenvectors of " + name(), dims());
    CoreTensorImpl *vals =
        new CoreTensorImpl("Eigenvalues of " + name(), {dims()[0]});

    vecs->copy(this);

    const size_t n = dims()[0];
    double dwork;
    C_DSYEV('V', 'U', n, vecs->data().data(), n, vals->data().data(), &dwork,
            -1);
    size_t lwork = (size_t)dwork;
    double *work = new double[lwork];
    C_DSYEV('V', 'U', n, vecs->data().data(), n, vals->data().data(), work,
            lwork);
    delete[] work;

    // If descending is required, the canonical order must be reversed
    // Sort is stable
    if (order == DescendingEigenvalue)
    {
        double *Temp_sqrsp_col = new double[n];
        double w_Temp_sqrsp;

        const size_t half_n = n >> 1;
        for (size_t c = 0; c < half_n; c++)
        {

            // Swap eigenvectors
            C_DCOPY(n, vecs->data().data() + c, n, Temp_sqrsp_col, 1);
            C_DCOPY(n, vecs->data().data() + n - c - 1, n,
                    vecs->data().data() + c, n);
            C_DCOPY(n, Temp_sqrsp_col, 1, vecs->data().data() + n - c - 1, n);

            // Swap eigenvalues
            w_Temp_sqrsp = vals->data().data()[c];
            vals->data().data()[c] = vals->data().data()[n - c - 1];
            vals->data().data()[n - c - 1] = w_Temp_sqrsp;
        }

        delete[] Temp_sqrsp_col;
    }

    map<string, TensorImplPtr> result;
    result["eigenvectors"] = vecs;
    result["eigenvalues"] = vals;

    return result;
}

map<string, TensorImplPtr> CoreTensorImpl::geev(EigenvalueOrder order) const
{
    squareCheck(this, true);

    CoreTensorImpl *A = new CoreTensorImpl("A of " + name(), dims());
    CoreTensorImpl *vl = new CoreTensorImpl("u of " + name(), dims());
    CoreTensorImpl *vr = new CoreTensorImpl("v of " + name(), dims());
    CoreTensorImpl *lambda =
        new CoreTensorImpl("lambda of " + name(), {dims()[0]});
    CoreTensorImpl *lambdai =
        new CoreTensorImpl("lambda i of " + name(), {dims()[0]});

    A->copy(this);

    int n = static_cast<int>(dims()[0]);
    double dwork;
    int info;
    info = C_DGEEV('V', 'V', n, A->data().data(), n, lambda->data().data(),
                   lambdai->data().data(), vl->data().data(), n,
                   vr->data().data(), n, &dwork, -1);
    int lwork = (int)dwork;
    vector<double> work(static_cast<size_t>(lwork));
    info = C_DGEEV('V', 'V', n, A->data().data(), n, lambda->data().data(),
                   lambdai->data().data(), vl->data().data(), n,
                   vr->data().data(), n, work.data(), lwork);

    if (info != 0)
    {
        throw std::runtime_error("CoreTensorImpl::geev: LAPACK call failed");
    }

    if (order == DescendingEigenvalue)
    {
        throw std::runtime_error("Unable to order descending");
    }

    map<string, TensorImplPtr> result;
    result["lambda"] = lambda;
    result["lambda i"] = lambdai;
    result["u"] = vl;
    result["v"] = vr;

    return result;
}

map<string, TensorImplPtr> CoreTensorImpl::gesvd() const
{
    rankCheck(2, this, true);

    int m = static_cast<int>(dim(0));
    int n = static_cast<int>(dim(1));
    int nsigma = m < n ? m : n;
    int lwork = m < n ? 5 * n : 5 * m;
    char jobvt = 'A';
    char jobu = 'A';

    CoreTensorImpl *U = new CoreTensorImpl(
        "U", {static_cast<size_t>(m), static_cast<size_t>(m)});
    CoreTensorImpl *V = new CoreTensorImpl(
        "V", {static_cast<size_t>(n), static_cast<size_t>(n)});
    CoreTensorImpl *Sigma =
        new CoreTensorImpl("Sigma", {static_cast<size_t>(nsigma)});
    vector<double> work(lwork, 0);

    // Make a copy of this's data into A since it will be destroyed in the
    // LAPACK call.
    vector<double> A(data().begin(), data().end());

    int info =
        C_DGESVD(jobu, jobvt, n, m, A.data(), n, Sigma->data().data(),
                 V->data().data(), n, U->data().data(), m, work.data(), lwork);

    if (info != 0)
    {
        throw std::runtime_error("CoreTensorImpl::gesvd: LAPACK call failed");
    }

    map<string, TensorImplPtr> results;
    results["U"] = U;
    results["V"] = V;
    results["Sigma"] = Sigma;

    return results;
}

// TensorImplPtr CoreTensorImpl::cholesky() const
//{
//    ThrowNotImplementedException;
//}
//
// map<string, TensorImplPtr> CoreTensorImpl::lu() const
//{
//    ThrowNotImplementedException;
//}
// map<string, TensorImplPtr> CoreTensorImpl::qr() const
//{
//    ThrowNotImplementedException;
//}
//
// TensorImplPtr CoreTensorImpl::cholesky_inverse() const
//{
//    ThrowNotImplementedException;
//}

TensorImplPtr CoreTensorImpl::inverse() const
{
    squareCheck(this, true);

    size_t n = dim(0);
    CoreTensorImpl *inverted = new CoreTensorImpl(name() + " Inverse", dims());
    vector<double> &a = inverted->data();

    memcpy(a.data(), data().data(), sizeof(double) * n * n);

    vector<int> ipiv(n, 0);
    int err = C_DGETRF(static_cast<int>(n), static_cast<int>(n), a.data(),
                       static_cast<int>(n), ipiv.data());

    if (err != 0)
    {
        if (err < 0)
        {
            ambit::print("CoreTensorImpl::inverse: C_DGETRF: argument %d has "
                         "invalid parameter.\n",
                         -err);
        }
        if (err > 0)
        {
            ambit::print("CoreTensorImpl::inverse: C_DGETRF: the (%d, %d) "
                         "element of the factor U or L is zero, inverse could "
                         "not be computed.\n",
                         err, err);
        }
        throw std::runtime_error(
            "CoreTensorImpl::inverse: C_DGETRF failed. See output.");
    }

    vector<double> work(n * n);
    err = C_DGETRI(static_cast<int>(n), a.data(), static_cast<int>(n),
                   ipiv.data(), work.data(), static_cast<int>(n * n));

    if (err != 0)
    {
        if (err < 0)
        {
            ambit::print("CoreTensorImpl::inverse: C_DGETRI: argument %d has "
                         "invalid parameter.\n",
                         -err);
        }
        if (err > 0)
        {
            ambit::print("CoreTensorImpl::inverse: C_DGETRI: the (%d, %d) "
                         "element of the factor U or L is zero, inverse could "
                         "not be computed.\n",
                         err, err);
        }
        throw std::runtime_error(
            "CoreTensorImpl::inverse: C_DGETRI failed. See output.");
    }
    return inverted;
}

TensorImplPtr CoreTensorImpl::power(double alpha, double condition) const
{
    // this call will ensure squareness
    map<string, TensorImplPtr> diag = syev(AscendingEigenvalue);

    size_t n = diag["eigenvalues"]->dims()[0];
    double *a =
        dynamic_cast<CoreTensorImplPtr>(diag["eigenvalues"])->data().data();
    double *a1 =
        dynamic_cast<CoreTensorImplPtr>(diag["eigenvectors"])->data().data();
    double *a2 = new double[n * n];

    memcpy(a2, a1, sizeof(double) * n * n);

    double max_a = (std::fabs(a[n - 1]) > std::fabs(a[0]) ? std::fabs(a[n - 1])
                                                          : std::fabs(a[0]));
    int remain = 0;
    for (size_t i = 0; i < n; i++)
    {

        if (alpha < 0.0 && fabs(a[i]) < condition * max_a)
            a[i] = 0.0;
        else
        {
            a[i] = pow(a[i], alpha);
            if (std::isfinite(a[i]))
            {
                remain++;
            }
            else
            {
                a[i] = 0.0;
            }
        }

        C_DSCAL(n, a[i], a2 + (i * n), 1);
    }

    CoreTensorImpl *powered =
        new CoreTensorImpl(name() + "^" + std::to_string(alpha), dims());

    C_DGEMM('T', 'N', n, n, n, 1.0, a2, n, a1, n, 0.0, powered->data_.data(),
            n);

    delete[] a2;

    // Need to manually delete the tensors in the diag map
    for (auto &el : diag)
    {
        delete el.second;
    }

    return powered;
}

void CoreTensorImpl::iterate(
    const function<void(const vector<size_t> &, double &)> &func)
{
    vector<size_t> indices(rank(), 0);
    vector<size_t> addressing(rank(), 1);

    // form addressing array
    for (int n = static_cast<int>(rank() - 2); n >= 0; --n)
    {
        addressing[n] = addressing[n + 1] * dim(n + 1);
    }

    const size_t nelem = numel();
    const size_t nrank = rank();
    for (size_t n = 0; n < nelem; ++n)
    {
        size_t d = n;
        for (size_t k = 0; k < nrank; ++k)
        {
            indices[k] = d / addressing[k];
            d = d % addressing[k];
        }

        func(indices, data_[n]);
    }
}

void CoreTensorImpl::citerate(
    const function<void(const vector<size_t> &, const double &)> &func) const
{
    vector<size_t> indices(rank(), 0);

    const size_t nelem = numel();
    const size_t nrank = rank();
    for (size_t n = 0; n < nelem; ++n)
    {
        size_t d = n;
        for (size_t k = 0; k < nrank; ++k)
        {
            indices[k] = d / addressing()[k];
            d = d % addressing()[k];
        }

        func(indices, data_[n]);
    }
}
} // namespace ambit
