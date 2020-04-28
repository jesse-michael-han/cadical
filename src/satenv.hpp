#include <pybind11/pybind11.h>

#include "internal.hpp"

namespace CaDiCaL
{
  class SatEnv
  {
    std::vector<unsigned> nv_to_v;
    Internal internal;
    CLIndices render (); // this sets nv_to_v
    std::tuple<CLIndices, double, bool> step (int l_ex); // this uses nv_to_v to translate external, simpified literal l_ex
    void reset ();
  };
}
