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
 * along with ambit; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <sys/stat.h>

#include "core/core.h"
#include "disk/disk.h"
#include "indices.h"
#include "tensorimpl.h"

#include <ambit/print.h>
#include <ambit/tensor.h>

#include "globals.h"

#include <ambit/timer.h>

// include header files to specific tensor types supported.
#if defined(HAVE_CYCLOPS)
#include "cyclops/cyclops.h"
#endif

namespace ambit
{

namespace settings
{

namespace
{

int ninitialized = 0;
}

int nprocess = 1;

int rank = 0;

bool debug = false;

size_t memory = 1 * 1024 * 1024 * 1024;

#if defined(HAVE_CYCLOPS)
const bool distributed_capable = true;
#else
const bool distributed_capable = false;
#endif

bool timers = false;
} // namespace settings

namespace
{

void common_initialize(int /*argc*/, char *const * /*argv*/)
{
    if (settings::ninitialized != 0)
    {
        throw std::runtime_error(
            "ambit::initialize: Ambit has already been initialized.");
    }

    settings::ninitialized++;

    timer::initialize();

    // Set the scratch path for disk files
    const char *scratch_env = std::getenv("TENSOR_SCRATCH");
    if (scratch_env != nullptr)
    {
        std::string scratch_str(scratch_env);
        Tensor::set_scratch_path(scratch_str);
    }
    else
    {
        Tensor::set_scratch_path(".");
    }
}
} // namespace

int initialize(int argc, char **argv)
{
    common_initialize(argc, argv);

#if defined(HAVE_CYCLOPS)
    return cyclops::initialize(argc, argv);
#else
    return 0;
#endif
}

void finalize()
{
    if (settings::ninitialized == 0)
    {
        throw std::runtime_error(
            "ambit::finalize: Ambit has already been finalized.");
    }

    settings::ninitialized--;

#if defined(HAVE_CYCLOPS)
    cyclops::finalize();
#endif

    timer::report();
    timer::finalize();
}

void barrier()
{
#if defined(HAVE_MPI)
    MPI_Barrier(globals::communicator);
#endif
}

string Tensor::scratch_path__ = ".";

Tensor::Tensor(shared_ptr<TensorImpl> tensor) : tensor_(std::move(tensor)) {}

Tensor Tensor::build(TensorType type, const string &name, const Dimension &dims)
{
    if (settings::ninitialized == 0)
    {
        throw std::runtime_error(
            "ambit::Tensor::build: Ambit has not been initialized.");
    }

    ambit::timer::timer_push("Tensor::build");

    Tensor newObject;

    if (type == AgnosticTensor)
    {
#if defined(HAVE_CYCLOPS)
        type = DistributedTensor;
#else
        type = CoreTensor;
#endif
    }
    switch (type)
    {
    case CoreTensor:
        newObject.tensor_.reset(new CoreTensorImpl(name, dims));
        break;

    case DiskTensor:
        newObject.tensor_.reset(new DiskTensorImpl(name, dims));
        break;

    case DistributedTensor:
#if defined(HAVE_CYCLOPS)
        newObject.tensor_.reset(new cyclops::CyclopsTensorImpl(name, dims));
#else
        throw std::runtime_error(
            "Tensor::build: Unable to construct distributed tensor object");
#endif

        break;

    default:
        throw std::runtime_error(
            "Tensor::build: Unknown parameter passed into 'type'.");
    }

    ambit::timer::timer_pop();

    return newObject;
}

Tensor Tensor::clone(TensorType type) const
{
    if (type == CurrentTensor)
        type = this->type();
    Tensor current = Tensor::build(type, name(), dims());
    current.copy(*this);
    return current;
}

void Tensor::reshape(const Dimension &dims) { tensor_->reshape(dims); }

void Tensor::resize(const Dimension &dims, bool trim)
{
    tensor_->resize(dims, trim);
}

void Tensor::copy(const Tensor &other) { tensor_->copy(other.tensor_.get()); }

Tensor::Tensor() {}

bool Tensor::is_set() { return static_cast<bool>(tensor_); }

void Tensor::reset() { tensor_.reset(); }

TensorType Tensor::type() const { return tensor_->type(); }

std::string Tensor::name() const { return tensor_->name(); }

void Tensor::set_name(const string &name) { tensor_->set_name(name); }

const Dimension &Tensor::dims() const { return tensor_->dims(); }

size_t Tensor::dim(size_t ind) const { return tensor_->dim(ind); }

size_t Tensor::rank() const { return tensor_->dims().size(); }

size_t Tensor::numel() const { return tensor_->numel(); }

void Tensor::print(FILE *fh, bool level, string const &format,
                   int maxcols) const
{
    tensor_->print(fh, level, format, maxcols);
}

LabeledTensor Tensor::operator()(const string &indices) const
{
    return LabeledTensor(*this, indices::split(indices));
}

SlicedTensor Tensor::operator()(const IndexRange &range) const
{
    return SlicedTensor(*this, range);
}

SlicedTensor Tensor::operator()() const
{
    IndexRange range;
    for (size_t ind = 0L; ind < rank(); ind++)
    {
        range.push_back({0L, dim(ind)});
    }
    return SlicedTensor(*this, range);
}

std::vector<double> &Tensor::data() { return tensor_->data(); }

const std::vector<double> &Tensor::data() const { return tensor_->data(); }

double &Tensor::at(const std::vector<size_t> &indices)
{
    return tensor_->at(indices);
}

const double &Tensor::at(const std::vector<size_t> &indices) const
{
    return tensor_->at(indices);
}

Tensor Tensor::cat(std::vector<Tensor> const, int dim)
{
    ThrowNotImplementedException;
}

double Tensor::norm(int type) const
{
    timer::timer_push("Tensor::norm");
    auto result = tensor_->norm(type);
    timer::timer_pop();
    return result;
}
void Tensor::zero()
{
    timer::timer_push("Tensor::zero");
    tensor_->scale(0.0);
    timer::timer_pop();
}

void Tensor::scale(double beta)
{
    timer::timer_push("Tensor::scale");
    tensor_->scale(beta);
    timer::timer_pop();
}

void Tensor::set(double alpha)
{
    timer::timer_push("Timer::set");
    tensor_->set(alpha);
    timer::timer_pop();
}

void Tensor::iterate(
    const std::function<void(const std::vector<size_t> &, double &)> &func)
{
    timer::timer_push("Tensor::iterate");
    tensor_->iterate(func);
    timer::timer_pop();
}

void Tensor::citerate(const std::function<void(const std::vector<size_t> &,
                                               const double &)> &func) const
{
    timer::timer_push("Tensor::citerate");
    tensor_->citerate(func);
    timer::timer_pop();
}

std::tuple<double, std::vector<size_t>> Tensor::max() const
{
    timer::timer_push("Tensor::max");
    auto result = tensor_->max();
    timer::timer_pop();

    return result;
}

tuple<double, vector<size_t>> Tensor::min() const
{
    timer::timer_push("Tensor::min");
    auto result = tensor_->min();
    timer::timer_pop();

    return result;
}

map<string, Tensor> Tensor::map_to_tensor(const map<string, TensorImplPtr> &x)
{
    map<string, Tensor> result;

    for (auto iter : x)
    {
        result.insert(
            make_pair(iter.first, Tensor(shared_ptr<TensorImpl>(iter.second))));
    }
    return result;
}

map<string, Tensor> Tensor::syev(EigenvalueOrder order) const
{
    timer::timer_push("Tensor::syev");
    auto result = map_to_tensor(tensor_->syev(order));
    timer::timer_pop();
    return result;
}

map<string, Tensor> Tensor::geev(EigenvalueOrder order) const
{
    timer::timer_push("Tensor::geev");
    auto result = map_to_tensor(tensor_->geev(order));
    timer::timer_pop();
    return result;
}

std::map<std::string, Tensor> Tensor::gesvd() const
{
    return map_to_tensor(tensor_->gesvd());
}

// Tensor Tensor::cholesky() const
//{
//    return Tensor(shared_ptr<TensorImpl>(tensor_->cholesky()));
//}
//
// std::map<std::string, Tensor> Tensor::lu() const
//{
//    return map_to_tensor(tensor_->lu());
//}
//
// std::map<std::string, Tensor> Tensor::qr() const
//{
//    return map_to_tensor(tensor_->qr());
//}
//
// Tensor Tensor::cholesky_inverse() const
//{
//    return Tensor(shared_ptr<TensorImpl>(tensor_->cholesky_inverse()));
//}

Tensor Tensor::inverse() const
{
    return Tensor(shared_ptr<TensorImpl>(tensor_->inverse()));
}

Tensor Tensor::power(double alpha, double condition) const
{
    return Tensor(shared_ptr<TensorImpl>(tensor_->power(alpha, condition)));
}

void Tensor::contract(const Tensor &A, const Tensor &B, const Indices &Cinds,
                      const Indices &Ainds, const Indices &Binds,
                      std::shared_ptr<TensorImpl> &A2,
                      std::shared_ptr<TensorImpl> &B2,
                      std::shared_ptr<TensorImpl> &C2, double alpha,
                      double beta)
{
    if (ambit::settings::debug)
    {
        ambit::print("    #: " + std::to_string(beta) + " " + name() + "[" +
                     indices::to_string(Cinds) +
                     "] = " + std::to_string(alpha) + " " + A.name() + "[" +
                     indices::to_string(Ainds) + "] * " + B.name() + "[" +
                     indices::to_string(Binds) + "]\n");
    }

    timer::timer_push("#: " + std::to_string(beta) + " " + name() + "[" +
                      indices::to_string(Cinds) +
                      "] = " + std::to_string(alpha) + " " + A.name() + "[" +
                      indices::to_string(Ainds) + "] * " + B.name() + "[" +
                      indices::to_string(Binds) + "]");

    tensor_->contract(A.tensor_.get(), B.tensor_.get(), Cinds, Ainds, Binds, A2,
                      B2, C2, alpha, beta);

    timer::timer_pop();
}
void Tensor::contract(const Tensor &A, const Tensor &B, const Indices &Cinds,
                      const Indices &Ainds, const Indices &Binds, double alpha,
                      double beta)
{
    if (ambit::settings::debug)
    {
        ambit::print("    #: " + std::to_string(beta) + " " + name() + "[" +
                     indices::to_string(Cinds) +
                     "] = " + std::to_string(alpha) + " " + A.name() + "[" +
                     indices::to_string(Ainds) + "] * " + B.name() + "[" +
                     indices::to_string(Binds) + "]\n");
    }

    timer::timer_push("#: " + std::to_string(beta) + " " + name() + "[" +
                      indices::to_string(Cinds) +
                      "] = " + std::to_string(alpha) + " " + A.name() + "[" +
                      indices::to_string(Ainds) + "] * " + B.name() + "[" +
                      indices::to_string(Binds) + "]");

    tensor_->contract(A.tensor_.get(), B.tensor_.get(), Cinds, Ainds, Binds,
                      alpha, beta);

    timer::timer_pop();
}
void Tensor::permute(const Tensor &A, const Indices &Cinds,
                     const Indices &Ainds, double alpha, double beta)
{
    if (ambit::settings::debug)
    {
        ambit::print("    P: " + name() + "[" + indices::to_string(Cinds) +
                     "] = " + A.name() + "[" + indices::to_string(Ainds) +
                     "]\n");
    }

    timer::timer_push("P: " + name() + "[" + indices::to_string(Cinds) +
                      "] = " + A.name() + "[" + indices::to_string(Ainds) +
                      "]");

    tensor_->permute(A.tensor_.get(), Cinds, Ainds, alpha, beta);

    timer::timer_pop();
}
void Tensor::slice(const Tensor &A, const IndexRange &Cinds,
                   const IndexRange &Ainds, double alpha, double beta)
{
    timer::timer_push("Tensor::slice");

    tensor_->slice(A.tensor_.get(), Cinds, Ainds, alpha, beta);

    timer::timer_pop();
}
void Tensor::gemm(const Tensor &A, const Tensor &B, bool transA, bool transB,
                  size_t nrow, size_t ncol, size_t nzip, size_t ldaA,
                  size_t ldaB, size_t ldaC, size_t offA, size_t offB,
                  size_t offC, double alpha, double beta)
{
    timer::timer_push("Tensor::gemm");
    tensor_->gemm(A.tensor_.get(), B.tensor_.get(), transA, transB, nrow, ncol,
                  nzip, ldaA, ldaB, ldaC, offA, offB, offC, alpha, beta);

    timer::timer_pop();
}

bool Tensor::operator==(const Tensor &other) const
{
    return tensor_ == other.tensor_;
}

bool Tensor::operator!=(const Tensor &other) const
{
    return tensor_ != other.tensor_;
}

void save(Tensor t, const std::string &filename, bool overwrite)
{
    // check if file exists or not
    struct stat buf;
    if (stat(filename.c_str(), &buf) == 0)
    {
        if (overwrite)
        {
            // delete the file
            if (remove(filename.c_str()) != 0)
            {
                std::string msg = "Error when deleting " + filename;
                perror(msg.c_str());
            }
        }
        else
        {
            std::string error = "File " + filename + " already exists.";
            throw std::runtime_error(error);
        }
    }
    // create the file
    std::ofstream out(filename.c_str(), std::ios_base::binary);
    // write tensor to file
    write_tensor_to_file(t, out);
    // close the file
    out.close();
}

void load(Tensor &t, const std::string &filename)
{
    // check if file exists or not
    std::ifstream in(filename.c_str(), std::ios_base::binary);
    if (!in.good())
    {
        std::string error = "File " + filename + " does not exist.";
        throw std::runtime_error(error);
    }
    // read tensor from file
    read_tensor_from_file(t, in);
    // close the file
    in.close();
}

Tensor load_tensor(const std::string &filename)
{
    Tensor t;
    load(t, filename);
    return t;
}

void write_tensor_to_file(Tensor t, std::ofstream &out)
{
    // 1. write the tensor name
    auto name = t.name();
    size_t size = name.size();
    out.write(reinterpret_cast<char *>(&size), sizeof(size_t));
    out.write(&name[0], size);

    // 2. write the rank and the size of each dimension
    size_t rank = t.rank();
    out.write(reinterpret_cast<char *>(&rank), sizeof(size_t));
    for (size_t m = 0; m < rank; m++)
    {
        size_t dim = t.dim(m);
        out.write(reinterpret_cast<char *>(&dim), sizeof(size_t));
    }

    // 3. write the size of the date and the data.
    size_t data_size = t.numel();
    out.write(reinterpret_cast<char *>(&data_size), sizeof(size_t));
    const std::vector<double> &data = t.data();
    out.write(reinterpret_cast<const char *>(&data[0]),
              data_size * sizeof(double));
}

void read_tensor_from_file(Tensor &t, std::ifstream &in)
{
    // 1. read the tensor name
    std::string name;
    size_t name_size;
    in.read(reinterpret_cast<char *>(&name_size), sizeof(size_t));
    name.resize(name_size);
    in.read(&name[0], name_size);

    // 2. read the rank and the size of each dimension
    size_t rank = 0;
    in.read(reinterpret_cast<char *>(&rank), sizeof(size_t));
    std::vector<size_t> dims(rank, 0);
    for (size_t m = 0; m < rank; m++)
    {
        size_t dim;
        in.read(reinterpret_cast<char *>(&dim), sizeof(size_t));
        dims[m] = dim;
    }

    // 3. allocate tensor or resize existing one
    if (t.is_set() == false)
    {
        t = Tensor::build(CoreTensor, name, dims);
    }
    else
    {
        t.set_name(name);
        t.resize(dims);
    }

    // 4. read the data
    size_t data_size;
    in.read(reinterpret_cast<char *>(&data_size), sizeof(size_t));
    std::vector<double> &data = t.data();
    in.read(reinterpret_cast<char *>(&data[0]), data_size * sizeof(double));
}

} // namespace ambit
