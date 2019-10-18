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
    const int hidden_size = 512;
    const int desire_size = 8;
    const int output_size = 384 + 385 + 385 + 58 + hidden_size;
    std::vector<float> outputbuffer;
    std::vector<float> desirebuffer;
    std::vector<float> hidden;

    SNPEModel snpe;
    std::string path;
public:
    Model(std::string path) : path(path),
        outputbuffer(output_size),
        desirebuffer(desire_size, 0.0f),
        hidden(hidden_size, 0.0f),
        snpe(path, outputbuffer.data(), output_size) {
        snpe.addDesire(desirebuffer.data(), desire_size);
        snpe.addRecurrent(hidden.data(), hidden_size);
    }

    Model(const Model& m): path(m.path),
        outputbuffer(output_size),
        desirebuffer(desire_size, 0.0f),
        hidden(hidden_size, 0.0f),
        snpe(path, outputbuffer.data(), output_size) {
        snpe.addDesire(desirebuffer.data(), desire_size);
        snpe.addRecurrent(hidden.data(), hidden_size);
    }

    ~Model() = default;

    bn::ndarray eval(bn::ndarray input) {
        snpe.execute(reinterpret_cast<float*>(input.get_data()));
        Py_intptr_t shape[1] = { output_size };
        bn::ndarray result = bn::zeros(1, shape, bn::dtype::get_builtin<float>());
        std::copy(outputbuffer.begin(), outputbuffer.end(), reinterpret_cast<float*>(result.get_data()));
        return result;
    }    
};

BOOST_PYTHON_MODULE(pysnpe) {
    bn::initialize();
    bp::class_< Model >("SNPEModel", bp::init<std::string>())
      .def("eval", &Model::eval);
}
