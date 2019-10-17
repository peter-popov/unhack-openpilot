#include "snpemodel.h"

#include "boost/python/numpy.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <iomanip>

namespace bp = boost::python;
namespace bn = boost::python::numpy;


class Model {
    const int output_size = 384+385+385+58+512;
    std::vector<float> outputbuffer;
    SNPEModel snpe;
    std::string path;
public:
    Model(std::string path) : path(path),
        outputbuffer(output_size),
        snpe(path, outputbuffer.data(), output_size) {
    }

    Model(const Model& m): path(m.path),
        outputbuffer(output_size),
        snpe(path, outputbuffer.data(), output_size) {
    }

    ~Model() = default;

    bn::ndarray eval(bn::ndarray input) {
        snpe.execute(reinterpret_cast<float*>(input.get_data()));
        Py_intptr_t shape[1] = { output_size };
        bn::ndarray result = bn::zeros(1, shape, bn::dtype::get_builtin<float>());
        std::copy(outputbuffer.begin(), outputbuffer.end(), reinterpret_cast<float*>(result.get_data()));
    }    
};

BOOST_PYTHON_MODULE(pysnpe) {
    bn::initialize();
    bp::class_< Model >("SNPEModel", bp::init<std::string>())
      .def("eval", &Model::eval);
}
