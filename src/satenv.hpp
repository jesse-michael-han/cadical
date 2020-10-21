#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cadical.hpp"
#include "internal.hpp"

namespace CaDiCaL
{
  struct StepResult {
    CLIndices obs;
    double reward;
    bool is_terminal;
  };
  
  class SatEnv
  {
    bool init_dimacs_flag = false;
    std::vector<unsigned> nv_to_v;
    Solver* solver;
  public:
    CLIndices render (); // this sets nv_to_v
    StepResult step (int l_ex); // this uses nv_to_v to translate external, simpified literal l_ex
    void reset ();
    bool is_terminal = false;
    const char * init_dimacs(const char* path);
    void reinit (const char * path);
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
                              .def("reinit", &CaDiCaL::SatEnv::reinit)
                              .def("reset", &CaDiCaL::SatEnv::reset);

                            py::class_<CaDiCaL::CLIndices>(m, "CLIndices")
                              .def(py::init())
                              .def_readwrite("n_vars", &CaDiCaL::CLIndices::n_vars)
                              .def_readwrite("n_clauses", &CaDiCaL::CLIndices::n_clauses)
                              .def_readwrite("C_idxs", &CaDiCaL::CLIndices::C_idxs)
                              .def_readwrite("L_idxs", &CaDiCaL::CLIndices::L_idxs);

                            py::class_<CaDiCaL::StepResult>(m, "StepResult")
                              .def(py::init())
                              .def_readonly("obs", &CaDiCaL::StepResult::obs)
                              .def_readonly("reward", &CaDiCaL::StepResult::reward)
                              .def_readonly("is_terminal", &CaDiCaL::StepResult::is_terminal);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
