#pragma once

#ifndef _torch_INCLUDED
#define _torch_INCLUDED

#include <vector>
#include <torch/script.h>
#undef LOG

namespace CaDiCaL
{
  class CLIndices {
  public:
  unsigned n_vars;
  unsigned n_clauses;
  std::vector<int> C_idxs; // TODO(jesse, March 18 2020, 04:36 PM): do this more efficiently with a libtorch array
  std::vector<int> L_idxs;
  CLIndices () = default;
  CLIndices (unsigned &x, unsigned &y, std::vector<int> &z, std::vector<int> &w): n_vars(x), n_clauses(y), C_idxs(z), L_idxs(w) {};

  void set_n_vars (unsigned const &nv) {n_vars = nv;};
  void set_n_clauses (unsigned const &nc) {n_clauses = nc;};
  void push_back (int const &c_idx, int const &l_idx)
  { C_idxs.push_back(c_idx); L_idxs.push_back(l_idx); };
    void dump (FILE* f = stdout);
};

  class GNN1 {
  public:
    torch::jit::script::Module module;
    bool CUDA_FLAG;
    torch::Tensor get_logits(CLIndices &CL_idxs);
    torch::Tensor operator()(CLIndices &CL_idxs) {return get_logits(CL_idxs);};
    std::string MODEL_PATH;
    GNN1() = default;
    void init_model(const char* model_path, int seed, bool use_gpu = false);
  };
}
#endif
