#include <boost/python.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <tensor/tensor.h>

using namespace boost::python;
using namespace tensor;

LabeledTensor tensor_getitem(Tensor& v, const std::string& indx)
{
    return v[indx];
}

BOOST_PYTHON_MODULE (pytensor)
{
    enum_<TensorType>("TensorType", "docstring")
            .value("kCurrent", kCurrent)
            .value("kCore", kCore)
            .value("kDisk", kDisk)
            .value("kDistributed", kDistributed)
            .value("kAgnostic", kAgnostic);

    enum_<EigenvalueOrder>("EigenvalueOrder", "docstring")
            .value("kAscending", kAscending)
            .value("kDescending", kDescending);

    class_<Dimension>("Dimension")
            .def(vector_indexing_suite<Dimension>());

    class_<std::pair<size_t, size_t> >("SizeTPair")
            .def_readwrite("first", &std::pair<size_t, size_t>::first)
            .def_readwrite("second", &std::pair<size_t, size_t>::second);

    class_<IndexRange>("IndexRange")
            .def(vector_indexing_suite<IndexRange>());

    class_<Indices>("Indices")
            .def(vector_indexing_suite<Indices>());

    class_<std::vector<double> >("DoubleVector")
            .def(vector_indexing_suite<std::vector<double> >());

    typedef std::vector<double>& (Tensor::*data1)();
//    typedef Tensor (Tensor::*build1)(TensorType, const std::string&, const Dimension&);

    class_<Tensor>("Tensor")
            .def("build", &Tensor::build)
            .staticmethod("build")
            .def("build_from", &Tensor::build_from)
            .staticmethod("build_from")
            .add_property("type", &Tensor::type, "docstring")
            .add_property("name", &Tensor::name, &Tensor::set_name, "docstring")
            .add_property("dims",
                          make_function(&Tensor::dims, return_internal_reference<>()), "docstring")
            .def("dim", &Tensor::dim, "docstring")
            .add_property("rank", &Tensor::dim, "docstring")
            .add_property("numel", &Tensor::numel, "docstring")
//            .def("data", make_function((std::vector<double>& (Tensor::*))&Tensor::data, return_internal_reference<>()), "docstring")
            .def("data", data1(&Tensor::data), return_internal_reference<>())
            .def("scale", &Tensor::scale)
            .def("permute", &Tensor::permute)
            .def("slice", &Tensor::slice)
            .def("contract", &Tensor::contract)
            .def("syev", &Tensor::syev)
            .def("geev", &Tensor::geev)
            .def("svd", &Tensor::svd)
            .def("cholesky", &Tensor::cholesky)
            .def("lu", &Tensor::lu)
            .def("qr", &Tensor::qr)
            .def("cholesky_inverse", &Tensor::cholesky_inverse)
            .def("inverse", &Tensor::inverse)
            .def("power", &Tensor::power)
            .def("print", &Tensor::print)
            .def("__getitem__", &tensor_getitem);

    class_<LabeledTensor>("LabeledTensor", no_init)
            .def(self += self)
            .def(self -= self)
            .def(self *= double())
            .def(self /= double());
}
