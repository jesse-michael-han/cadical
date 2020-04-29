#include <pybind11/pybind11.h>

#include "cadical.hpp"
#include "internal.hpp"

namespace CaDiCaL
{
  class SatEnv
  {
    bool init_dimacs_flag = false;
    std::vector<unsigned> nv_to_v;
    Solver* solver;
  public:
    CLIndices render (); // this sets nv_to_v
    std::tuple<CLIndices, double, bool> step (int l_ex); // this uses nv_to_v to translate external, simpified literal l_ex
    void reset ();
    const char * init_dimacs(const char* path);
    SatEnv (const char * path);
    ~SatEnv ();
  };
}

namespace py = pybind11;

PYBIND11_MODULE(satenv, m){
                            py::class_<CaDiCaL::SatEnv>(m, "SatEnv")
                            .def(py::init<const char *>())
                              .def("render", &CaDiCaL::SatEnv::render)
                              .def("step", &CaDiCaL::SatEnv::step)
                              .def("reset", &CaDiCaL::SatEnv::reset);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
