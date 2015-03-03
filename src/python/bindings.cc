#include <boost/python.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <ambit/tensor.h>
#include <../tensor/indices.h>

#include <boost/shared_ptr.hpp>

using namespace boost::python;
using namespace ambit;

/** @brief Type that allows for registration of conversions from
 *         Python iterable types.
 */
struct iterable_converter
{
    /** @note Registers converter from a Python iterable type to the
     *  provided type.
     */
    template<typename Container>
    iterable_converter&
    from_python()
    {
        boost::python::converter::registry::push_back(&iterable_converter::convertible,
                                                      &iterable_converter::construct<Container>,
                                                      boost::python::type_id<Container>());

        // support chaining
        return *this;
    }

    /// @brief Check if PyObject is iterable
    static void* convertible(PyObject* object)
    {
        return PyObject_GetIter(object) ? object : nullptr;
    }

    /** @brief Convert iterable PyObject to C++ container type.
     *
     * Container concept requirements:
     *
     *   * Container::value_type is CopyConstructable.
     *   * Container can be constructed and populated with two iterators.
     *     i.e. Container(begin, end)
     */
    template<typename Container>
    static void construct(PyObject* object,
                          boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        namespace python = boost::python;

        // Object is borrowed reference, so create a handle indictating it is
        // borrowed for proper reference counting
        python::handle<> handle(python::borrowed(object));

        // Obtain a handle to the memory block that the converter has allocated
        // for the C++ type.
        typedef python::converter::rvalue_from_python_storage<Container> storage_type;

        void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

        typedef python::stl_input_iterator<typename Container::value_type> iterator;

        // Allocate the C++ type into the converter's memory block, and assign
        // its handle to the converter's convertible variable. The C++
        // container is populated by passing the begin and end iterators of
        // the python object to the container's constructor.
        new (storage) Container(iterator(python::object(handle)), // begin
                                iterator());                      // end
        data->convertible = storage;
    }
};

void initialize_random(Tensor& A1)
{
    size_t numel1 = A1.numel();
    aligned_vector<double>& A1v = A1.data();
    for (size_t ind = 0L; ind < numel1; ind++) {
        double randnum = double(std::rand())/double(RAND_MAX);
        A1v[ind] = randnum;
    }
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(tensor_print_ov, Tensor::print, 0, 4)

BOOST_PYTHON_MODULE (pyambit)
{

    // Register iterable conversions.
    iterable_converter()
            .from_python<std::vector<size_t>>()      // same as a Dimension object
            .from_python<std::vector<std::vector<size_t>>>();

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

    class_<std::pair<size_t, size_t>>("SizeTPair")
            .def_readwrite("first", &std::pair<size_t, size_t>::first)
            .def_readwrite("second", &std::pair<size_t, size_t>::second);

    class_<std::vector<double>>("DoubleVector")
            .def(vector_indexing_suite<std::vector<double>>());

    class_<aligned_vector<double>>("AlignedDoubleVector")
            .def(vector_indexing_suite<aligned_vector<double>>());

    class_<IndexRange>("IndexRange")
            .def(vector_indexing_suite<IndexRange>());

    class_<Indices>("Indices")
            .def(vector_indexing_suite<Indices>())
            .def("split", &indices::split)
            .staticmethod("split")
            .def("permutation_order", &indices::permutation_order)
            .staticmethod("permutation_order");

    typedef aligned_vector<double>& (Tensor::*data1)();
    typedef const Indices& (LabeledTensor::*idx)() const;

    class_<LabeledTensor>("LabeledTensor", no_init)
            .def(init<Tensor, const std::vector<std::string>&, double>())
            .add_property("factor", &LabeledTensor::factor, "docstring")
            .add_property("indices", make_function(idx(&LabeledTensor::indices), return_value_policy<copy_const_reference>()));

    class_<Tensor>("Tensor", no_init)
            .def("build", &Tensor::build)
            .staticmethod("build")
            .add_property("type", &Tensor::type, "docstring")
            .add_property("name", &Tensor::name, &Tensor::set_name, "docstring")
            .add_property("dims",
                          make_function(&Tensor::dims, return_internal_reference<>()), "docstring")
            .def("dim", &Tensor::dim, "docstring")
            .add_property("rank", &Tensor::rank, "docstring")
            .add_property("numel", &Tensor::numel, "docstring")
            .def("data", make_function(data1(&Tensor::data), return_internal_reference<>()))
            .def("scale", &Tensor::scale)
            .def("permute", &Tensor::permute)
            .def("slice", &Tensor::slice)
            .def("contract", &Tensor::contract)
            .def("syev", &Tensor::syev)
            .def("power", &Tensor::power)
            .def("norm", &Tensor::norm)
            .def("zero", &Tensor::zero)
            .def("printf", &Tensor::print,tensor_print_ov());

    def("initialize_random", &initialize_random);
}
