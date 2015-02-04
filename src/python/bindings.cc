#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <tensor/tensor.h>

using namespace boost::python;
using namespace tensor;

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

    typedef Tensor (Tensor::*build1)(TensorType, const std::string&, const Dimension&);
    typedef Tensor (Tensor::*build2)(TensorType, const Tensor&);

    class_<Tensor>("Tensor")
            .def("build", build1(Tensor::build), "docstring")
            .staticmethod("build")
            .def("build", build2(Tensor::build), "docstring")
            .staticmethod("build")
            .def("type", &Tensor::type, "docstring")
            .def("name", &Tensor::name, "docstring");
}
