#include <boost/python.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

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
    template <typename Container> iterable_converter &from_python()
    {
        boost::python::converter::registry::push_back(
            &iterable_converter::convertible,
            &iterable_converter::construct<Container>,
            boost::python::type_id<Container>());

        // support chaining
        return *this;
    }

    /// @brief Check if PyObject is iterable
    static void *convertible(PyObject *object)
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
    template <typename Container>
    static void
    construct(PyObject *object,
              boost::python::converter::rvalue_from_python_stage1_data *data)
    {
        namespace python = boost::python;

        // Object is borrowed reference, so create a handle indictating it is
        // borrowed for proper reference counting
        python::handle<> handle(python::borrowed(object));

        // Obtain a handle to the memory block that the converter has allocated
        // for the C++ type.
        typedef python::converter::rvalue_from_python_storage<Container>
            storage_type;

        void *storage = reinterpret_cast<storage_type *>(data)->storage.bytes;

        typedef python::stl_input_iterator<typename Container::value_type>
            iterator;

        // Allocate the C++ type into the converter's memory block, and assign
        // its handle to the converter's convertible variable. The C++
        // container is populated by passing the begin and end iterators of
        // the python object to the container's constructor.
        new (storage) Container(iterator(python::object(handle)), // begin
                                iterator());                      // end
        data->convertible = storage;
    }
};

dict tensor_array_interface(Tensor ten)
{
    dict rv;

    rv["shape"] = boost::python::tuple(ten.dims());
    rv["data"] = boost::python::make_tuple((long)ten.data().data(), false);

    // Type
    // std::string typestr = is_big_endian() ? ">" : "<";
    std::string typestr = "<";
    std::stringstream sstr;
    sstr << (int)sizeof(double);
    typestr += "f" + sstr.str();
    rv["typestr"] = typestr;

    return rv;
}

void initialize_wrapper() { ambit::initialize(0, nullptr); }

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(tensor_print_ov, Tensor::print, 0, 4)

BOOST_PYTHON_MODULE(pyambit)
{

    // Register iterable conversions.
    iterable_converter()
        .from_python<std::vector<size_t>>() // same as a Dimension object
        .from_python<std::vector<double>>() // need to expressly make a double
                                            // converter
        .from_python<std::vector<std::vector<size_t>>>() // same as IndexRange
        .from_python<std::vector<std::string>>();

    class_<std::vector<Tensor>>("TensorVector")
        .def(vector_indexing_suite<std::vector<Tensor>>());

    class_<std::map<std::string, Tensor>>("TensorMap")
        .def(map_indexing_suite<std::map<std::string, Tensor>>());

    class_<Dimension>("Dimension").def(vector_indexing_suite<Dimension>());

    class_<std::pair<size_t, size_t>>("SizeTPair")
        .def_readwrite("first", &std::pair<size_t, size_t>::first)
        .def_readwrite("second", &std::pair<size_t, size_t>::second);

    class_<std::vector<double>>("DoubleVector")
        .def(vector_indexing_suite<std::vector<double>>());

    class_<IndexRange>("IndexRange").def(vector_indexing_suite<IndexRange>());

    class_<std::vector<Indices>>("IndicesVector")
        .def(vector_indexing_suite<std::vector<Indices>>());

    // Typedefs
    enum_<TensorType>("TensorType", "docstring")
        .value("CurrentTensor", CurrentTensor)
        .value("CoreTensor", CoreTensor)
        .value("DiskTensor", DiskTensor)
        .value("DistributedTensor", DistributedTensor)
        .value("AgnosticTensor", AgnosticTensor);

    enum_<EigenvalueOrder>("EigenvalueOrder", "docstring")
        .value("AscendingEigenvalue", AscendingEigenvalue)
        .value("DescendingEigenvalue", DescendingEigenvalue);

    class_<Indices>("Indices")
        .def(vector_indexing_suite<Indices>())
        .def("split", &indices::split)
        .staticmethod("split")
        .def("permutation_order", &indices::permutation_order)
        .staticmethod("permutation_order")
        .def("determine_contraction_result_from_indices",
             &indices::determine_contraction_result_from_indices)
        .staticmethod("determine_contraction_result_from_indices");

    typedef const Indices &(LabeledTensor::*idx)() const;
    std::vector<double> &(Tensor::*data)() = &Tensor::data;

    class_<LabeledTensor>("ILabeledTensor", no_init)
        .def(init<Tensor, const std::vector<std::string> &, double>())
        .add_property("factor", &LabeledTensor::factor, "docstring")
        .add_property(
            "indices",
            make_function(idx(&LabeledTensor::indices),
                          return_value_policy<copy_const_reference>()))
        .def("dim_by_index", &LabeledTensor::dim_by_index);

    class_<Tensor>("ITensor", no_init)
        .def("build", &Tensor::build)
        .staticmethod("build")
        .add_property("dtype", &Tensor::type, "docstring")
        .add_property("name", &Tensor::name, &Tensor::set_name, "docstring")
        .add_property(
            "dims", make_function(&Tensor::dims, return_internal_reference<>()),
            "docstring")
        .def("dim", &Tensor::dim, "docstring")
        .add_property("rank", &Tensor::rank, "docstring")
        .add_property("numel", &Tensor::numel, "docstring")
        .def("data", data, return_value_policy<reference_existing_object>())
        .def("scale", &Tensor::scale)
        .def("permute", &Tensor::permute)
        .def("slice", &Tensor::slice)
        .def("contract", &Tensor::contract)
        .def("syev", &Tensor::syev)
        .def("geev", &Tensor::geev)
        .def("power", &Tensor::power)
        .def("norm", &Tensor::norm)
        .def("zero", &Tensor::zero)
        .def("copy", &Tensor::copy)
        .def("min", &Tensor::min)
        .def("max", &Tensor::max)
        .def("printf", &Tensor::print, tensor_print_ov())
        .def("reset", &Tensor::reset)
        .def("__array_interface__", tensor_array_interface);

    def("initialize", initialize_wrapper);
    def("finalize", ambit::finalize);
}
