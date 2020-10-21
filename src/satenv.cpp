#include "satenv.hpp"
#include <iostream>
#include <numeric>

namespace CaDiCaL
{

  const char * SatEnv::init_dimacs(const char* path){

      auto x = 0;
      auto y = 2;
      auto err = solver->read_dimacs (path, x, y);
      if (err) {
        throw std::runtime_error(err);
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

  void SatEnv::reinit (const char * path) {
    solver->~Solver();
    solver = new Solver ();
    init_dimacs(path);
    nv_to_v = std::vector<unsigned>(solver->internal->max_var);
    std::iota (std::begin(nv_to_v), std::end(nv_to_v), 0);
    is_terminal = false;
  }
  
  SatEnv::~SatEnv () {
    delete solver;
  }

  CLIndices SatEnv::render () {
    if (is_terminal) { return CLIndices (); }
    auto foo = solver->internal->buildCLIndices();
    auto CL_idxs = std::get<0>(foo);
    auto new_nv_to_v = std::get<1>(foo);
    nv_to_v = new_nv_to_v;
    return CL_idxs;
  }

  StepResult SatEnv::step (int l_ex) {
    if (is_terminal) throw std::runtime_error("environment state is terminal, cannot assign variable");
    double reward = - 1.0/(solver->internal->max_var);
    int l_in = sign(l_ex) * (nv_to_v[abs(l_ex) - 1] + 1);
    solver->internal->search_assume_decision(l_in);
    bool status_after_propagation = solver->internal->propagate();
    if (!status_after_propagation) { reward = 1/(max(pow(solver->internal->analyze2(), 2.0), 1.0)); is_terminal = true; }
    else { if (solver->internal->satisfied()) is_terminal = true; }
    // auto num_clauses = solver->internal->clauses.size();
    // bool NO_LEARNED_FLAG = solver->internal->clauses.size() == num_clauses;
    // if (!NO_LEARNED_FLAG) throw std::runtime_error("learned a clause, uh-oh");
    StepResult result = {render(), reward, is_terminal};
    return result;
  }

  void SatEnv::reset () {

    solver->internal->backtrack(0);
    // nv_to_v.clear();
    nv_to_v.clear();
    nv_to_v = std::vector<unsigned>(solver->internal->max_var);
    std::iota (std::begin(nv_to_v), std::end(nv_to_v), 0);
    // for (auto i = 0; i < solver->internal->max_var; i++) {
    //   nv_to_v.push_back(i);
    // }
    is_terminal = false;
  }
}
