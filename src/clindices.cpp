#include "internal.hpp"

namespace CaDiCaL
{
  std::tuple<CLIndices, std::vector<unsigned>> Internal::buildCLIndices()
  {
    // simplify();
    int n_vars = max_var;

    unsigned n_clauses = 0;

    // unsigned max_n_nodes_cells = 10000000; // TODO(jesse): don't hardcode this
    // unsigned max_lclause_size = 1000; // TODO(jesse): don't hardcode this

    std::vector<int> v_to_nv;
    std::vector<unsigned> nv_to_v;

    std::vector<bool> assigned(n_vars, false);

    // int trail_limit = (trail_lim.size() == 0 ? trail.size() : trail_lim[0]);

    // for (int i = 0; i < trail_limit; ++i) {
    //   assigned[var(trail[i])] = true;
    // }
    for (auto l : trail) {
      // std::cout << "NEXT VARAIBLE INDEX ON TRAIL: " << var(trail[i]) << "\n";
      assigned[vidx(l)-1] = true;
    }

    for (int v = 0; v < n_vars; v++) {
      if (assigned[v] // || isEliminated(v) -- minisat doesn't implement variable elimination?
          ) {
        v_to_nv.push_back(-1);
      } else {
        v_to_nv.push_back(nv_to_v.size());
        nv_to_v.push_back(v_to_nv.size() - 1); // storing shifts for adjacency graph minimization
        // suppose that a new variable index is returned--- then one can recover the original variable index by slicing nv_to_v
      }
    }

    auto CL_idxs = CLIndices();

    auto new_n_vars = nv_to_v.size();
    CL_idxs.set_n_vars(new_n_vars);

    int push_count = 0;

    // the solver is responsible for maintaining state to translate shifted variables
    // i.e. this function emits a possibly compressed adjacency matrix; this is all the variable selection heuristic ever sees, and when the result is returned, it is shifted again
    // note: for now, do not pass learned clauses to the heuristic

    bool SIZE_EXCEEDED = false;
  
    auto traverse_clause = [&](Clause & clause) {
                             if (clause.garbage) {
                               return;
                             } else if (2 * n_vars + n_clauses + push_count > 10000000) {
                               SIZE_EXCEEDED = true;
                               return;
                               // } else if (clause.learnt() && (unsigned) clause.size() > max_lclause_size) {
                               //   return;
                             }
                               else if (clause.redundant && clause.size > 1000) {
                                 return;
                               }                             
                             // else if (clause.redundant) {
                             //   return;
                             // }

                             // else if (satisfied(clause)) {
                             //   return;
                             // }

                             else {
                               for (auto lit : clause) {
                                 auto v_idx = vidx(lit)-1;
                                 if (v_idx >= n_vars) {throw std::runtime_error("var too big");}
                                 int new_v_idx = v_to_nv[v_idx];
                                 if (new_v_idx !=  -1 && new_v_idx >= (int) new_n_vars) {
                                   std::cout << "V_IDX: " << v_idx << " NEW_V_IDX: " << new_v_idx << " NEW_N_VAR: " << new_n_vars << "\n";
                                 }
                                 if (new_v_idx != -1) {
                                   CL_idxs.push_back(n_clauses, sign(lit) ? new_v_idx : (new_v_idx + new_n_vars) );
                                   push_count++;
                                 }
                               }

                               
                               // for (int arg_idx = 0; arg_idx < clause.size; ++arg_idx) {
                               //   Lit lit = clause[arg_idx];
                               //   Var v_idx   = var(lit);
                               //   if (v_idx >= n_vars) { throw std::runtime_error("var too big!"); }
                               //   int new_v_idx  = v_to_nv[v_idx]; // get compressed variable index
                               //   if (new_v_idx != -1 && new_v_idx >= new_n_vars) // comparisons between unsigned ints are the root of all evil
                               //     {std::cout << "V_IDX: " << v_idx << " NEW_V_IDX: " << new_v_idx << " NEW_N_VAR: " << new_n_vars << "\n";}
                               
                               //   // if (new_v_idx >= new_n_vars) {throw std::runtime_error("new v_idx too big!"); }
                               //   if (new_v_idx != -1) {
                               //     CL_idxs.push_back(n_clauses, sign(lit) ? new_v_idx : (new_v_idx + new_n_vars) );
                               //     push_count++;
                               //     // args.add_c_idxs(n_clauses);
                               //     // args.add_l_idxs(sign(lit) ? (nv + args.n_vars()) : nv); // TODO(jesse): fix; do something to cl_idxs
                               //   }
                               // }
                               n_clauses++;
                               return;
                             }
                           };


    // unsigned long c_idx = 0;
    // // for (int c_idx = 0; c_idx < clauses.size(); ++c_idx)
    // while (c_idx < clauses.size())
    //   {
    //     traverse_clause(ca[clauses[c_idx]]);
    //     if (SIZE_EXCEEDED) {
    //       throw TooBigError();
    //     }
    //     c_idx ++;
    //   }
    for (auto cls : clauses) {
      traverse_clause(*cls); // TODO(jesse): implement cutoff limit -- assuming learned clauses are pushed back, throw a flag when a learned clause is encountered and begin counting from there
    }

    // auto c_idx2 = 0;

    // sort(learnts, ClauseSize_lt(ca));  

    // while (c_idx2 < learnts.size())
    //   {
    //     traverse_clause(ca[learnts[c_idx2]]);
    //     if (SIZE_EXCEEDED) {
    //       break;
    //     }
    //     c_idx2 ++;
    //   }
    // printf("NUMBER OF LEARNT CLAUSES SERIALIZED %d\n", c_idx2);
    // printf("THE BIGGEST CLAUSE: %d\n", c_idx + c_idx2);

    CL_idxs.set_n_clauses(n_clauses);
    return std::tuple<CLIndices, std::vector<unsigned>> {CL_idxs, nv_to_v};    
  }

  torch::Tensor GNN1::get_logits(CLIndices &CL_idxs) {
    long n_cells = CL_idxs.C_idxs.size();

    auto C_indices = torch::from_blob(CL_idxs.C_idxs.data(), {n_cells}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kLong);
    auto L_indices = torch::from_blob(CL_idxs.L_idxs.data(), {n_cells}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kLong);
    auto indices = torch::stack({C_indices, L_indices});

    auto values = torch::ones({n_cells}).to(torch::kFloat32);
    int64_t n_clauses = CL_idxs.n_clauses;
    int64_t n_lits = CL_idxs.n_vars * 2;
    std::vector<int64_t> sizes = {n_clauses, n_lits};

    auto G = at::sparse_coo_tensor(indices, values, sizes);
    std::vector<torch::jit::IValue> inputs = {G};

    // auto outputs = module.forward(inputs).toTuple();
    // auto V_drat_logits = outputs -> elements()[0].toTensor().squeeze();
    auto outputs = module.forward(inputs).toTensor();
    auto V_logits = outputs.squeeze();

    // V_drat_logits = V_drat_logits.view({1, V_drat_logits.size(0)});
    // auto V_core_logits = outputs[1];
    return V_logits;
  }
}
