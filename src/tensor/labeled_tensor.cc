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

#include <list>
#include <algorithm>
#include <numeric>
#include <ambit/tensor.h>
#include "tensorimpl.h"
#include "indices.h"

namespace ambit
{

LabeledTensor::LabeledTensor(Tensor T, const Indices &indices, double factor)
    : T_(T), indices_(indices), factor_(factor)
{
    if (T_.rank() != indices.size())
        throw std::runtime_error("Labeled tensor does not have correct number "
                                 "of indices for underlying tensor's rank");
}

void LabeledTensor::set(const LabeledTensor &to)
{
    T_ = to.T_;
    indices_ = to.indices_;
    factor_ = to.factor_;
}

size_t LabeledTensor::dim_by_index(const string &idx) const
{
    // determine location of idx in indices_
    Indices::const_iterator location =
        std::find(indices_.begin(), indices_.end(), idx);
    if (location == indices_.end())
        throw std::runtime_error("Index not found: " + idx);

    size_t position = std::distance(indices_.begin(), location);
    return T().dim(position);
}

void LabeledTensor::operator=(const LabeledTensor &rhs)
{
    if (T() == rhs.T())
        throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Permuted tensors do not have same rank");
    T_.permute(rhs.T(), indices_, rhs.indices(), rhs.factor(), 0.0);
}

void LabeledTensor::operator+=(const LabeledTensor &rhs)
{
    if (T() == rhs.T())
        throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Permuted tensors do not have same rank");
    T_.permute(rhs.T(), indices_, rhs.indices(), rhs.factor(), 1.0);
}

void LabeledTensor::operator-=(const LabeledTensor &rhs)
{
    if (T() == rhs.T())
        throw std::runtime_error("Self assignment is not allowed.");
    if (T_.rank() != rhs.T().rank())
        throw std::runtime_error("Permuted tensors do not have same rank");
    T_.permute(rhs.T(), indices_, rhs.indices(), -rhs.factor(), 1.0);
}

LabeledTensorContraction LabeledTensor::operator*(const LabeledTensor &rhs)
{
    return LabeledTensorContraction(*this, rhs);
}

LabeledTensorAddition LabeledTensor::operator+(const LabeledTensor &rhs)
{
    return LabeledTensorAddition(*this, rhs);
}

LabeledTensorAddition LabeledTensor::operator-(const LabeledTensor &rhs)
{
    return LabeledTensorAddition(*this, -rhs);
}

namespace
{

LabeledTensor tensor_product_get_temp_AB(const LabeledTensor &A,
                                         const LabeledTensor &B)
{
    std::vector<Indices> AB_indices =
        indices::determine_contraction_result(A, B);
    const Indices &A_fix_idx = AB_indices[1];
    const Indices &B_fix_idx = AB_indices[2];
    Dimension dims;
    Indices indices;

    for (size_t i = 0; i < A_fix_idx.size(); ++i)
    {
        dims.push_back(A.T().dim(i));
        indices.push_back(A_fix_idx[i]);
    }
    for (size_t i = 0; i < B_fix_idx.size(); ++i)
    {
        dims.push_back(B.T().dim(i));
        indices.push_back(B_fix_idx[i]);
    }

    Tensor T =
        Tensor::build(A.T().type(), A.T().name() + " * " + B.T().name(), dims);
    return T(indices::to_string(indices));
}
}

void LabeledTensor::contract(const LabeledTensorContraction &rhs,
                             bool zero_result, bool add)
{
    size_t nterms = rhs.size();
    std::vector<size_t> perm(nterms);
    std::vector<size_t> best_perm(nterms);
    std::iota(perm.begin(), perm.end(), 0);
    std::pair<double, double> best_cpu_memory_cost(1.0e200, 1.0e200);

    do
    {
        std::pair<double, double> cpu_memory_cost =
            rhs.compute_contraction_cost(perm);
        if (cpu_memory_cost.first < best_cpu_memory_cost.first)
        {
            best_perm = perm;
            best_cpu_memory_cost = cpu_memory_cost;
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    // at this point 'best_perm' should be used to perform contraction in
    // optimal order.

    LabeledTensor A = rhs[best_perm[0]];
    int maxn = int(nterms) - 2;
    for (int n = 0; n < maxn; ++n)
    {
        LabeledTensor B = rhs[best_perm[n + 1]];

        std::vector<Indices> AB_indices =
            indices::determine_contraction_result(A, B);
        const Indices &AB_common_idx = AB_indices[0];
        const Indices &A_fix_idx = AB_indices[1];
        const Indices &B_fix_idx = AB_indices[2];
        Dimension dims;
        Indices indices;

        for (size_t i = 0; i < AB_common_idx.size(); ++i)
        {
            // If a common index is also found in the rhs it's a Hadamard index
            if (std::find(this->indices().begin(), this->indices().end(),
                          AB_common_idx[i]) != this->indices().end())
            {
                dims.push_back(A.dim_by_index(AB_common_idx[i]));
                indices.push_back(AB_common_idx[i]);
            }
        }

        for (size_t i = 0; i < A_fix_idx.size(); ++i)
        {
            dims.push_back(A.dim_by_index(A_fix_idx[i]));
            indices.push_back(A_fix_idx[i]);
        }
        for (size_t i = 0; i < B_fix_idx.size(); ++i)
        {
            dims.push_back(B.dim_by_index(B_fix_idx[i]));
            indices.push_back(B_fix_idx[i]);
        }

        Tensor tAB = Tensor::build(A.T().type(),
                                   A.T().name() + " * " + B.T().name(), dims);

        tAB.contract(A.T(), B.T(), indices, A.indices(), B.indices(),
                     A.factor() * B.factor(), 0.0);

        A.set(LabeledTensor(tAB, indices, 1.0));
    }
    const LabeledTensor &B = rhs[best_perm[nterms - 1]];

    T_.contract(A.T(), B.T(), indices(), A.indices(), B.indices(),
                add ? A.factor() * B.factor() : -A.factor() * B.factor(),
                zero_result ? 0.0 : 1.0);
}

void LabeledTensor::operator=(const LabeledTensorContraction &rhs)
{
    contract(rhs, true, true);
}

void LabeledTensor::operator+=(const LabeledTensorContraction &rhs)
{
    contract(rhs, false, true);
}

void LabeledTensor::operator-=(const LabeledTensorContraction &rhs)
{
    contract(rhs, false, false);
}

void LabeledTensor::operator=(const LabeledTensorAddition &rhs)
{
    T_.zero();
    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind)
    {
        if (T_ == rhs[ind].T())
            throw std::runtime_error("Self assignment is not allowed.");
        if (T_.rank() != rhs[ind].T().rank())
            throw std::runtime_error("Permuted tensors do not have same rank");
        T_.permute(rhs[ind].T(), indices_, rhs[ind].indices(),
                   rhs[ind].factor(), 1.0);
    }
}

void LabeledTensor::operator+=(const LabeledTensorAddition &rhs)
{
    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind)
    {
        if (T_ == rhs[ind].T())
            throw std::runtime_error("Self assignment is not allowed.");
        if (T_.rank() != rhs[ind].T().rank())
            throw std::runtime_error("Permuted tensors do not have same rank");
        T_.permute(rhs[ind].T(), indices_, rhs[ind].indices(),
                   rhs[ind].factor(), 1.0);
    }
}

void LabeledTensor::operator-=(const LabeledTensorAddition &rhs)
{
    for (size_t ind = 0, end = rhs.size(); ind < end; ++ind)
    {
        if (T_ == rhs[ind].T())
            throw std::runtime_error("Self assignment is not allowed.");
        if (T_.rank() != rhs[ind].T().rank())
            throw std::runtime_error("Permuted tensors do not have same rank");
        T_.permute(rhs[ind].T(), indices_, rhs[ind].indices(),
                   -rhs[ind].factor(), 1.0);
    }
}

void LabeledTensor::operator*=(double scale) { T_.scale(scale); }

void LabeledTensor::operator/=(double scale) { T_.scale(1.0 / scale); }

LabeledTensorDistribution LabeledTensor::
operator*(const LabeledTensorAddition &rhs)
{
    return LabeledTensorDistribution(*this, rhs);
}

void LabeledTensor::operator=(const LabeledTensorDistribution &rhs)
{
    T_.zero();

    for (const LabeledTensor &B : rhs.B())
    {
        *this += const_cast<LabeledTensor &>(rhs.A()) *
                 const_cast<LabeledTensor &>(B);
    }
}

void LabeledTensor::operator+=(const LabeledTensorDistribution &rhs)
{
    for (const LabeledTensor &B : rhs.B())
    {
        *this += const_cast<LabeledTensor &>(rhs.A()) *
                 const_cast<LabeledTensor &>(B);
    }
}

void LabeledTensor::operator-=(const LabeledTensorDistribution &rhs)
{
    for (const LabeledTensor &B : rhs.B())
    {
        *this -= const_cast<LabeledTensor &>(rhs.A()) *
                 const_cast<LabeledTensor &>(B);
    }
}

LabeledTensorDistribution LabeledTensorAddition::
operator*(const LabeledTensor &other)
{
    return LabeledTensorDistribution(other, *this);
}

LabeledTensorAddition &LabeledTensorAddition::operator*(double scalar)
{
    // distribute the scalar to each term
    for (LabeledTensor &T : tensors_)
    {
        T *= scalar;
    }

    return *this;
}

LabeledTensorAddition &LabeledTensorAddition::operator-()
{
    for (LabeledTensor &T : tensors_)
    {
        T *= -1.0;
    }

    return *this;
}

LabeledTensorContraction::operator double() const
{
    Tensor R = Tensor::build(tensors_[0].T().type(), "R", {});
    LabeledTensor lR(R, {}, 1.0);
    lR.contract(*this, true, true);

    Tensor C = Tensor::build(CoreTensor, "C", {});
    C.slice(R, {}, {});

    return C.data()[0];
}

pair<double, double> LabeledTensorContraction::compute_contraction_cost(
    const vector<size_t> &perm) const
{
#if 0
    printf("\n\n  Testing the cost of the contraction pattern: ");
    for (size_t p : perm)
        printf("[");
    for (size_t p : perm) {
        const LabeledTensor &ti = tensors_[p];
        printf(" %s] ", indices::to_string(ti.indices()).c_str());
    }
#endif

    map<string, size_t> indices_to_size;

    for (const LabeledTensor &ti : tensors_)
    {
        const Indices &indices = ti.indices();
        for (size_t i = 0; i < indices.size(); ++i)
        {
            indices_to_size[indices[i]] = ti.T().dim(i);
        }
    }

    double cpu_cost_total = 0.0;
    double memory_cost_max = 0.0;
    Indices first = tensors_[perm[0]].indices();
    for (size_t i = 1; i < perm.size(); ++i)
    {
        Indices second = tensors_[perm[i]].indices();
        std::sort(first.begin(), first.end());
        std::sort(second.begin(), second.end());
        Indices common, first_unique, second_unique;

        // cannot use common.begin() here, need to use back_inserter() because
        // common.begin() of an
        // empty vector is not a valid output iterator
        std::set_intersection(first.begin(), first.end(), second.begin(),
                              second.end(), back_inserter(common));
        std::set_difference(first.begin(), first.end(), second.begin(),
                            second.end(), back_inserter(first_unique));
        std::set_difference(second.begin(), second.end(), first.begin(),
                            first.end(), back_inserter(second_unique));

        double common_size = 1.0;
        for (const string &s : common)
            common_size *= indices_to_size[s];
        double first_size = 1.0;
        for (const string &s : first)
            first_size *= indices_to_size[s];
        double second_size = 1.0;
        for (const string &s : second)
            second_size *= indices_to_size[s];
        double first_unique_size = 1.0;
        for (const string &s : first_unique)
            first_unique_size *= indices_to_size[s];
        double second_unique_size = 1.0;
        for (const string &s : second_unique)
            second_unique_size *= indices_to_size[s];
        double result_size = first_unique_size + second_unique_size;

        Indices stored_indices(first_unique);
        stored_indices.insert(stored_indices.end(), second_unique.begin(),
                              second_unique.end());

        double cpu_cost = common_size * result_size;
        double memory_cost = first_size + second_size + result_size;
        cpu_cost_total += cpu_cost;
        memory_cost_max = std::max({memory_cost_max, memory_cost});

#if 0
        printf("\n  First indices        : %s", indices::to_string(first).c_str());
        printf("\n  Second indices       : %s", indices::to_string(second).c_str());

        printf("\n  Common indices       : %s (%.0f)", indices::to_string(common).c_str(), common_size);
        printf("\n  First unique indices : %s (%.0f)", indices::to_string(first_unique).c_str(), first_unique_size);
        printf("\n  Second unique indices: %s (%.0f)", indices::to_string(second_unique).c_str(), second_unique_size);

        printf("\n  CPU cost for this step    : %f.0", cpu_cost);
        printf("\n  Memory cost for this step : %f.0 = %f.0 + %f.0 + %f.0",
               memory_cost,
               first_size,
               second_size,
               result_size);

        printf("\n  Stored indices       : %s", indices::to_string(stored_indices).c_str());
#endif

        first = stored_indices;
    }

#if 0
    printf("\n  Total CPU cost                : %f.0", cpu_cost_total);
    printf("\n  Maximum memory cost           : %f.0", memory_cost_max);
#endif

    return std::make_pair(cpu_cost_total, memory_cost_max);
}

LabeledTensorDistribution::operator double() const
{
    Tensor R = Tensor::build(A_.T().type(), "R", {});

    for (size_t ind = 0L; ind < B_.size(); ind++)
    {

        R.contract(A_.T(), B_[ind].T(), {}, A_.indices(), B_[ind].indices(),
                   A_.factor() * B_[ind].factor(), 1.0);
    }

    Tensor C = Tensor::build(CoreTensor, "C", {});
    C.slice(R, {}, {});

    return C.data()[0];
}

LabeledSlicedTensor::LabeledSlicedTensor(Tensor T, const Indices &indices, const Indices &sliced_indices, double factor)
    : T_(T), indices_(indices), sliced_indices_(sliced_indices), factor_(factor)
{
    if (T_.rank() != indices.size())
        throw std::runtime_error("Labeled tensor does not have correct number "
                                 "of indices for underlying tensor's rank");
}

size_t LabeledSlicedTensor::dim_by_index(const string &idx) const
{
    // determine location of idx in indices_
    Indices::const_iterator location =
        std::find(indices_.begin(), indices_.end(), idx);
    if (location == indices_.end())
        throw std::runtime_error("Index not found: " + idx);

    size_t position = std::distance(indices_.begin(), location);
    return T().dim(position);
}

void LabeledSlicedTensor::operator=(const LabeledTensorContraction &rhs)
{
    contract(rhs, true, true);
}

void LabeledSlicedTensor::operator+=(const LabeledTensorContraction &rhs)
{
    contract(rhs, false, true);
}

void LabeledSlicedTensor::operator-=(const LabeledTensorContraction &rhs)
{
    contract(rhs, false, false);
}

void LabeledSlicedTensor::contract(const LabeledTensorContraction &rhs,
                             bool zero_result, bool add)
{
    size_t nterms = rhs.size();
    size_t slice_size = sliced_indices_.size();
    std::vector<size_t> slicing_axis(slice_size);
    for (size_t l = 0; l < slice_size; ++l) {
        auto it = std::find(indices_.begin(), indices_.end(), sliced_indices_[l]);
        if (it != indices_.end()) {
            slicing_axis[l] = std::distance(indices_.begin(), it);
        } else {
            throw std::runtime_error("Slicing indices do not exist in tensor indices.");
        }
    }
    bool permute_flag = false;
    for (size_t l = 0; l < slice_size; ++l) {
        if (slicing_axis[l] != l) {
            permute_flag = true;
            break;
        }
    }
    Tensor Ltp;
    Indices permuted_indices;
    Dimension slicing_dims;
    for (const string& s : sliced_indices_) {
        slicing_dims.push_back(dim_by_index(s));
    }

    if (permute_flag) {
        permuted_indices.insert(permuted_indices.end(), sliced_indices_.begin(), sliced_indices_.end());
        for (size_t l = 0, l_max = numdim(); l < l_max; ++l) {
            if (std::find(slicing_axis.begin(), slicing_axis.end(),l) == slicing_axis.end()) {
                permuted_indices.push_back(indices_[l]);
            }
        }
        Dimension dims;
        for (const string& s : permuted_indices) {
            dims.push_back(dim_by_index(s));
        }
        Ltp = Tensor::build(T().type(), T().name() + " permute", dims);
        Ltp.permute(T_, permuted_indices, indices_);
    } else {
        Ltp = T().clone();
        permuted_indices = indices_;
    }
    LabeledTensor Lt(Ltp, permuted_indices);

    std::vector<std::vector<bool>> need_slicing(nterms, std::vector<bool>(slice_size + 1));

    LabeledTensorContraction rhsp;
    for (size_t i = 0; i < nterms; ++i) {
        const LabeledTensor& A = rhs[i];
        const Indices& A_indices = A.indices();
        Indices gemm_indices;
        for (const string& s : A_indices) {
            auto it = std::find(sliced_indices_.begin(), sliced_indices_.end(), s);
            if (it != sliced_indices_.end()) {
                need_slicing[i][std::distance(sliced_indices_.begin(), it)] = true;
            } else {
                gemm_indices.push_back(s);
            }
        }
        Indices permuted_indices;
        for (size_t l = 0; l < slice_size; ++l) {
            if (need_slicing[i][l]) {
                permuted_indices.push_back(sliced_indices_[l]);
                need_slicing[i][slice_size] = true;
            }
        }
        if (permuted_indices.size() == 0) {
            rhsp.operator*(A);
        } else {
            permuted_indices.insert(permuted_indices.end(),gemm_indices.begin(),gemm_indices.end());
            Dimension dims;
            for (const string& s : permuted_indices) {
                dims.push_back(A.dim_by_index(s));
            }
            Tensor Atp = Tensor::build(A.T().type(), A.T().name() + " permute", dims);
            Atp.permute(A.T(), permuted_indices, A.indices());
            LabeledTensor At(Atp, permuted_indices);
            rhsp.operator*(At);
        }
    }

    const Indices& Lt_indices = Lt.indices();
    const Dimension& Lt_dims = Lt.T().dims();
    Dimension sub_dims;
    Indices sub_indices;
    sub_dims.insert(sub_dims.end(), Lt_dims.begin()+slice_size, Lt_dims.end());
    sub_indices.insert(sub_indices.end(), Lt_indices.begin()+slice_size, Lt_indices.end());
    Tensor Ltp_slice = Tensor::build(Lt.T().type(), Lt.T().name() + " slice", sub_dims);
    LabeledTensor Lt_slice(Ltp_slice, sub_indices);

    size_t num_slice = 1;
    for (size_t l = 0; l < slice_size; ++l) {
        num_slice *= slicing_dims[l];
    }
    LabeledTensorContraction rhs_slice;

    std::vector<Tensor> tensors;
    for (size_t i = 0; i < nterms; ++i) {
        const LabeledTensor& A = rhsp[i];
        if (need_slicing[i][slice_size]) {
            const Indices& A_indices = A.indices();
            const Dimension& A_dims = A.T().dims();
            Dimension dims;
            Indices indices;
            size_t count = 0;
            for (size_t l = 0; l < slice_size; ++l) {
                if (need_slicing[i][l]) {
                    count++;
                }
            }
            indices.insert(indices.end(), A_indices.begin()+count, A_indices.end());
            dims.insert(dims.end(), A_dims.begin()+count, A_dims.end());

            Tensor Atp = Tensor::build(A.T().type(), A.T().name() + " slice", dims);
            LabeledTensor At(Atp, indices);
            rhs_slice.operator*(At);
            tensors.push_back(Atp);
        } else {
            rhs_slice.operator*(A);
            tensors.push_back(A.T());
        }
    }

    Dimension current_slice(slice_size, 0);
    while (current_slice[0] < slicing_dims[0]) {
        size_t L_shift = 0, cur_jump = 1;
        size_t sub_numel = Lt_slice.T().numel();
        for (int i = slice_size - 1; i >= 0; --i) {
            L_shift += current_slice[i] * cur_jump;
            cur_jump *= slicing_dims[i];
        }
        L_shift *= sub_numel;
        std::vector<double>& Lt_slice_data = Ltp_slice.data();
        std::vector<double>& Lt_data = Ltp.data();
        Lt_slice_data.clear();
        Lt_slice_data.insert(Lt_slice_data.end(), Lt_data.begin()+L_shift, Lt_data.begin()+L_shift+sub_numel);

        for (size_t i = 0; i < nterms; ++i) {
            if (need_slicing[i][slice_size]) {
                size_t cur_shift = 0, cur_jump = 1;
                size_t sub_numel_A = tensors[i].numel();
                for (int l = slice_size - 1; l >= 0; --l) {
                    if (need_slicing[i][l]) {
                        cur_shift += current_slice[l] * cur_jump;
                        cur_jump *= slicing_dims[l];
                    }
                }
                cur_shift *= sub_numel_A;
                std::vector<double>& A_slice_data = tensors[i].data();
                const std::vector<double>& A_data = rhsp[i].T().data();
                A_slice_data.clear();
                A_slice_data.insert(A_slice_data.end(), A_data.begin()+cur_shift, A_data.begin()+cur_shift+sub_numel_A);
            }
        }

        Lt_slice.contract(rhs_slice, zero_result, add);

        const std::vector<double>& Ltc_slice_data = Lt_slice.T().data();
        for (size_t i = 0; i < sub_numel; ++i) {
            Lt_data[L_shift + i] = Ltc_slice_data[i];
        }

        for (int i = slice_size - 1; i >= 0; --i) {
            current_slice[i]++;
            if (current_slice[i] < slicing_dims[i]) {
                break;
            } else if (i != 0) {
                current_slice[i] = 0;
            }
        }
    }

    T_.permute(Ltp, indices_, permuted_indices);
}

}
