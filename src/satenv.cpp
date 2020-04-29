#include "satenv.hpp"
#include <iostream>

namespace CaDiCaL
{

  int main () {
    std::cout << "hello world" << "\n";
    return 0;
  }

  const char * SatEnv::init_dimacs(const char* path){
      auto x = 0;
      auto y = 2;
      auto err = solver->read_dimacs (path, x, y);
      if (err) {
        throw std::runtime_error("parsing failed");
      }
      init_dimacs_flag = true;
      return err;
  };  

  SatEnv::SatEnv (const char * path) {
      solver = new Solver ();
      init_dimacs(path);
      nv_to_v = std::vector<unsigned>(solver->internal->max_var);
      std::iota (std::begin(nv_to_v), std::end(nv_to_v), 0);
  }
  
  SatEnv::~SatEnv () {
    delete solver;
  }

  CLIndices SatEnv::render () {
    auto [CL_idxs, new_nv_to_v] = solver->internal->buildCLIndices();
    nv_to_v = new_nv_to_v;
    return CL_idxs;
  }

  std::tuple<CLIndices, double, bool> SatEnv::step (int l_ex) {
    auto num_clauses = solver->internal->clauses.size();
    double reward = - 1.0/(solver->internal->max_var);        
    int l_in = nv_to_v[l_ex - 1] + 1;
    solver->internal->search_assume_decision(l_in);
    bool achieved_conflict = solver->internal->propagate();
    if (!achieved_conflict) { reward = 4.0 * 1/pow(solver->internal->analyze2(), 2.0); }
    bool NO_LEARNED_FLAG = solver->internal->clauses.size() == num_clauses;
    assert(NO_LEARNED_FLAG); // TODO(jesse): remove this
    return {render(), reward, !achieved_conflict};
  }

  void SatEnv::reset () {
    solver->internal->backtrack(0);
  }
}
