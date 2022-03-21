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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <ambit/tensor.h>
#include <../tensor/indices.h>

namespace py = pybind11;
using namespace pybind11::literals;

using namespace ambit;

py::dict tensor_array_interface(Tensor ten)
{
    py::dict rv;

    // Cast the NumPy shape vector
    py::tuple shape_tuple = py::cast(ten.dims());
    rv["shape"] = shape_tuple;
    rv["version"] = 3;
    rv["data"] = py::make_tuple((long)ten.data().data(), false);

    // Type
    // std::string typestr = is_big_endian() ? ">" : "<";
    std::string typestr = "<";
    std::stringstream sstr;
    sstr << (int)sizeof(double);
    typestr += "f" + sstr.str();
    rv["typestr"] = typestr;

    return rv;
}

// TODO: There's a lot of code duplication b/w these two parse_slices calls.
// When the rest of the code settles down, consolidate.
IndexRange parse_slices(const py::tuple& tuple, const Dimension& dim)
{
    IndexRange idx;
    if (tuple.size() != dim.size())
        throw std::exception();

    for (size_t i = 0; i < tuple.size(); ++i) {
        const auto& items = tuple[i];
        const auto slice = items.cast<py::slice>();
        try {
            slice.attr("step").cast<py::none>();
        } catch(...) {
            throw std::invalid_argument("Step syntax not allowed. Try the NumPy interface.");
        }
        size_t start, stop;
        try {
            start = slice.attr("start").cast<size_t>();
        } catch(...) {
            start = 0;
        }
        try {
            stop = slice.attr("stop").cast<size_t>();
        } catch(...) {
            stop = dim[i];
        }
        idx.push_back(std::vector<size_t> {start, stop});
    }
    return idx;
}

IndexRange parse_slices(const py::slice& slice, const Dimension& dim)
{
    IndexRange idx;
    // The below check should have been done in the Python version but wasn't.
    // TODO: Uncomment this check after the rest is corrected.
    /*
    if (1 != dim.size())
        throw std::exception();
    */
    
    try {
        slice.attr("step").cast<py::none>();
    } catch(...) {
        throw std::invalid_argument("Step syntax not allowed. Try the NumPy interface.");
    }
    size_t start, stop;
    try {
        start = slice.attr("start").cast<size_t>();
    } catch(...) {
        start = 0;
    }
    try {
        stop = slice.attr("stop").cast<size_t>();
    } catch(...) {
        stop = dim[0];
    }
    idx.push_back(std::vector<size_t> {start, stop});

    return idx;
}


void initialize_wrapper() { ambit::initialize(0, nullptr); }

PYBIND11_MODULE(pyambit, m)
{
    // Typedefs
    py::enum_<TensorType>(m, "TensorType", "docstring")
        .value("CurrentTensor", TensorType::CurrentTensor)
        .value("CoreTensor", TensorType::CoreTensor)
        .value("DiskTensor", TensorType::DiskTensor)
        .value("DistributedTensor", TensorType::DistributedTensor)
        .value("AgnosticTensor", TensorType::AgnosticTensor)
        .export_values();

    py::enum_<EigenvalueOrder>(m, "EigenvalueOrder", "docstring")
        .value("AscendingEigenvalue", EigenvalueOrder::AscendingEigenvalue)
        .value("DescendingEigenvalue", EigenvalueOrder::DescendingEigenvalue)
        .export_values();

    py::class_<Indices>(m, "Indices")
        .def_static("split", &indices::split)
        .def_static("permutation_order", &indices::permutation_order)
        .def_static("determine_contraction_result_from_indices",
             &indices::determine_contraction_result_from_indices);

    typedef const Indices &(LabeledTensor::*idx)() const;
    std::vector<double> &(Tensor::*data)() = &Tensor::data;

    py::class_<LabeledTensor>(m, "ILabeledTensor")
        .def(py::init<Tensor, const std::vector<std::string> &, double>(), "tensor"_a, "indices"_a, "factor"_a = 1)
        .def(py::init([](Tensor T, const std::string& indices, double factor) {
            std::vector<std::string> index_vector;
            for (auto& ch : indices) {
                index_vector.push_back(std::string(1, ch));
            }
            return std::make_unique<LabeledTensor>(T, index_vector, factor);
        }), "tensor"_a, "indices"_a, "factor"_a = 1)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def("__iadd__", [](LabeledTensor& s1, const LabeledTensorContraction& s2) { return s1 += s2; })
        .def("__isub__", [](LabeledTensor& s1, const LabeledTensorContraction& s2) { return s1 -= s2; })
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def("__mul__", [](const LabeledTensor& s1, const LabeledTensorAddition& s2) { return s1 * s2; })
        .def(float() * py::self)
        .def(py::self *= float())
        .def(py::self /= float())
        .def_property_readonly("factor", &LabeledTensor::factor, "docstring")
        .def_property_readonly("indices", idx(&LabeledTensor::indices), py::return_value_policy::copy)
        .def_property_readonly("tensor", &LabeledTensor::T)
        .def("dim_by_index", &LabeledTensor::dim_by_index)
        .def("set", &LabeledTensor::set);

    py::class_<LabeledTensorAddition>(m, "LabeledTensorAddition")
        .def(py::init<const LabeledTensor&, const LabeledTensor&>())
        .def(py::init<>())
        .def("__iter__", [](const LabeledTensorAddition& t) { return py::make_iterator(t.begin(), t.end()); }, py::keep_alive<0, 1>())
        .def(-py::self)
        .def("__add__", [](const LabeledTensorAddition& s1, const LabeledTensor& s2) { return s1 + s2; })
        .def("__iadd__", [](LabeledTensorAddition& s1, const LabeledTensor& s2) { return s1 += s2; })
        .def("__mul__", [](const LabeledTensorAddition& s1, const LabeledTensor& s2) { return s1 * s2; })
        .def(float() * py::self);

    py::class_<LabeledTensorContraction>(m, "LabeledTensorContraction")
        .def(py::init<const LabeledTensor&, const LabeledTensor&>())
        .def(py::init<>())
        .def("__float__", [](const LabeledTensorContraction& t) { return static_cast<double>(t); })
        .def("__len__", &LabeledTensorContraction::size)
        .def("__imul__", [](LabeledTensorContraction& s1, const LabeledTensor& s2) { return s1 *= s2; })
        .def(float() * py::self)
        .def("__getitem__", [](const LabeledTensorContraction& t, size_t i) { return t[i]; })
        .def("compute_contraction_cost", &LabeledTensorContraction::compute_contraction_cost);

    py::class_<LabeledTensorDistribution>(m, "LabeledTensorDistributive")
        .def(py::init<const LabeledTensor&, const LabeledTensorAddition&>())
        .def("__float__", [](const LabeledTensorDistribution& t) { return static_cast<double>(t); })
        .def_property_readonly("A", &LabeledTensorDistribution::A)
        .def_property_readonly("B", &LabeledTensorDistribution::B);

    py::class_<SlicedTensor>(m, "SlicedTensor")
        .def(py::init<Tensor, const IndexRange &, double>(), "T"_a, "range"_a, "factor"_a=1)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def_property_readonly("tensor", &SlicedTensor::T)
        .def_property_readonly("range", &SlicedTensor::range)
        .def_property_readonly("rank", &SlicedTensor::rank)
        .def_property_readonly("factor", &SlicedTensor::factor);

    typedef void (Tensor::*contract1)(const Tensor &A, const Tensor &B, const Indices &Cinds,
            const Indices &Ainds, const Indices &Binds, double alpha, double beta);

    py::class_<Tensor>(m, "Tensor")
    //py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        /*
         * The below buffer code is the direct pybind/NumPy interface.
         * Unfortunately, it doesn't play nicely with Py-side _array_interface_
        .def_buffer([](Tensor &t) -> py::buffer_info {
                // Taken from pybind11/buffer_info.h:c_strides
                auto shape = t.dims();
                auto ndim = shape.size();
                std::vector<ssize_t> strides(ndim, sizeof(double));
                if (ndim > 0)
                    for (size_t i = ndim - 1; i > 0; --i)
                        strides[i - 1] = strides[i] * shape[i];
                return py::buffer_info(
                    t.data().data(),
                    shape,
                    strides
                 );
         })
         */
        .def_static("build", &Tensor::build)
        .def_property_readonly("dtype", &Tensor::type, "docstring")
        .def_property("name", &Tensor::name, &Tensor::set_name, "docstring")
        .def_property_readonly("dims", &Tensor::dims, py::return_value_policy::reference_internal)
        .def("dim", &Tensor::dim, "docstring")
        .def_property_readonly("rank", &Tensor::rank, "docstring")
        .def_property_readonly("numel", &Tensor::numel, "docstring")
        .def("scale", &Tensor::scale)
        .def("permute", &Tensor::permute)
        .def("slice", &Tensor::slice)
        .def("contract", contract1(&Tensor::contract))
        .def("syev", &Tensor::syev)
        .def("geev", &Tensor::geev)
        .def("power", &Tensor::power)
        .def("norm", &Tensor::norm)
        .def("zero", &Tensor::zero)
        .def("copy", &Tensor::copy)
        .def("min", &Tensor::min)
        .def("max", &Tensor::max)
        .def("printf", [](const Tensor& self, bool level = true, std::string string = "%%11.6f", int maxcols = 5) {
                self.print(stdout, level, string, maxcols); })
        .def("reset", &Tensor::reset)
        .def("set", &Tensor::set)
        .def("__getitem__", [](const Tensor& t, const std::string& s) { return t(s); })
        .def("__getitem__", [](const Tensor& t, const IndexRange& indices) { return t(indices); })
        .def("__getitem__", [](const Tensor& t, const py::tuple& indices) { return t(parse_slices(indices, t.dims())); })
        .def("__getitem__", [](const Tensor& t, const py::slice& indices) { return t(parse_slices(indices, t.dims())); })
        .def("__setitem__", [](Tensor& t, const std::string& key, const LabeledTensor& s) { t(key) = s; })
        .def("__setitem__", [](Tensor& t, const std::string& key, const LabeledTensorAddition& s) { t(key) = s; })
        .def("__setitem__", [](Tensor& t, const std::string& key, const LabeledTensorContraction& s) { t(key) = s; })
        .def("__setitem__", [](Tensor& t, const std::string& key, const LabeledTensorDistribution& s) { t(key) = s; })
        .def("__setitem__", [](Tensor& t, const IndexRange& indices, const SlicedTensor& s) { t(indices) = s; })
        .def("__setitem__", [](Tensor& t, const py::tuple& indices, const SlicedTensor& s) { t(parse_slices(indices, t.dims())) = s ; })
        .def_property_readonly("__array_interface__", tensor_array_interface);

    m.def("initialize", initialize_wrapper);
    m.def("finalize", ambit::finalize);
}
