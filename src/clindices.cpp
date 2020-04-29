#include "internal.hpp"
#include <iostream>
// #include "torch/nn/parallel/data_parallel.h" // don't include this, as it's unneeded -- rely on the user to supply GPU availability in form of opts.gpu flag

namespace CaDiCaL
{

  void GNN1::init_model(const char* model_path, int seed, bool use_gpu)
  {
    // torch::manual_seed(seed);
    // MODEL_PATH = std::string(model_path);
    // CUDA_FLAG = false;
    // // CUDA_FLAG = true;
    // // module.to(at::kCUDA);    
    
    // if (use_gpu)
    // {
    //   CUDA_FLAG = true;
    //   module = torch::jit::load(MODEL_PATH, torch::kCUDA);
    // }
    // else
    // {
    //   module = torch::jit::load(MODEL_PATH);
    // }
    return;
  }

  // struct Clause_lt {
  //   bool operator() (Clause* c1, Clause* c2)
  //   {
  //     if (!c1 -> redundant && c2 -> redundant) {
  //       return true;
  //     }
  //     else
  //       {
  //         if (!c2 -> redundant && c1 -> redundant) {
  //           return false;
  //         }
  //         else {
  //           if (!c1 -> redundant) {
  //             return false;
  //           }
  //           else {
  //             return c1 -> glue < c2 -> glue;
  //           }
  //         }
  //       }
  //   }
  // };
  // struct Clause_rk {
  //   unsigned operator() (Clause *c1) {
  //     if (!c1 -> redundant) return 0;
  //     else {
  //       return c1 -> glue;
  //     }
  //   }
  // };

  void CLIndices::dump (FILE* f)
  {
    for (unsigned i = 0; i < C_idxs.size(); i++) {
      fprintf(f, "%d %d\n", C_idxs[i], L_idxs[i]);
    }
  };
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
    std::vector<bool> assigned_polarity(n_vars, false);

    // int trail_limit = (trail_lim.size() == 0 ? trail.size() : trail_lim[0]);

    // for (int i = 0; i < trail_limit; ++i) {
    //   assigned[var(trail[i])] = true;
    // }

    for (auto l : trail) {
      // std::cout << "NEXT VARAIBLE INDEX ON TRAIL: " << var(trail[i]) << "\n";
      assigned[vidx(l)-1] = true;
      assigned_polarity[vidx(l) - 1] = sign(l) == 1;
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

    int new_n_vars = nv_to_v.size();
    // std::cout << "NEW N VARS " << new_n_vars << "\n";
    // std::cout << "v_to_nv " << v_to_nv << "\n";
    CL_idxs.set_n_vars(new_n_vars);

    int push_count = 0;

    // the solver is responsible for maintaining state to translate shifted variables
    // i.e. this function emits a possibly compressed adjacency matrix; this is all the variable selection heuristic ever sees, and when the result is returned, it is shifted again

    bool SIZE_EXCEEDED_IRR = false;
    bool SIZE_EXCEEDED_RED = false;
    bool LEARNED_FLAG = false;    

    auto traverse_clause = [&](Clause & clause) {
                             if (clause.garbage) {
                               return;
                             } else if (2 * n_vars + n_clauses + push_count > (unsigned) opts.irrlim ) {
                               if (!LEARNED_FLAG) SIZE_EXCEEDED_IRR = true;
                               else SIZE_EXCEEDED_RED = true;
                               return;
                               // } else if (clause.learnt() && (unsigned) clause.size() > max_lclause_size) {
                               //   return;
                             }
                             else if (clause.redundant) {
                                 LEARNED_FLAG = true;
                                 // if ((double) clause.glue > averages.current.glue.slow) return;
                                 // if (!clause.keep) return;
                                 if (clause.glue > 5) return;
                               }
                             // else if (clause.redundant) {
                             //   return;
                             // }

                             // else if (satisfied(clause)) {
                             //   return;
                             // }

                             else {
                               std::vector<int> tmp_c_idx;
                               std::vector<int> tmp_l_idx;
                               
                               for (const auto &lit : clause) {
                                 
                                 auto v_idx = vidx(lit)-1;
                                 if (v_idx >= n_vars) {throw std::runtime_error("var too big");}
                                 int new_v_idx = v_to_nv[v_idx];
                                 if (new_v_idx !=  -1 && new_v_idx >= (int) new_n_vars) {
                                   std::cout << "V_IDX: " << v_idx << " NEW_V_IDX: " << new_v_idx << " NEW_N_VAR: " << new_n_vars << "\n";
                                   throw std::runtime_error("too many vars");
                                 }
                                 // if (n_clauses == 175) { std::cout << "V_IDX " <<  v_idx << "NEW_V_IDX " << new_v_idx << "NEW N VARS " << new_n_vars << "LIT " << lit << "\n"; }                                 
                                 if (new_v_idx != -1) {
                                   auto l_idx = (sign(lit) == 1) ? new_v_idx : (new_v_idx + new_n_vars);
                                   tmp_c_idx.push_back(n_clauses);
                                   tmp_l_idx.push_back(l_idx);
                                   // CL_idxs.push_back(n_clauses,  );
                                   // if (n_clauses == 175) { std::cout << "V_IDX " <<  v_idx << "NEW_V_IDX " << new_v_idx << "NEW N VARS " << new_n_vars << "LIT " << lit << "\n"; }                                 
                                   // push_count++;
                                   // std::cout << "PUSHED EDGE" << n_clauses << " " << (sign(lit) ? new_v_idx : (new_v_idx + new_n_vars)) << "\n" << "V_IDX : " << new_v_idx << "\n";
                                 }
                                 else { // clause is possibly satisfied
                                   auto pol = assigned_polarity[v_idx];
                                   if (pol == (sign(lit) == 1)) return;
                                 }

                               }

                               for (int i = 0; i < (int) tmp_c_idx.size(); i++) {
                                 CL_idxs.push_back(tmp_c_idx[i], tmp_l_idx[i]);
                                 push_count++;
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

    // rsort(clauses.begin(), clauses.end(), Clause_rk());

    // std::cout << "TRAVERSING CLAUSES\n";
    
    for (const auto & cls : clauses) {
      traverse_clause(*cls); // TODO(jesse): implement cutoff limit -- assuming learned clauses are pushed back, throw a flag when a learned clause is encountered and begin counting from there
      if (SIZE_EXCEEDED_IRR) {return {CLIndices(), {}};}
      if (SIZE_EXCEEDED_RED) {std::cout << "SIZE EXCEEDED BUT CONTINUING\n"; break; }
    }
    // std::cout << "CONTINUING\n";

    

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

  // torch::Tensor GNN1::get_logits(CLIndices &CL_idxs) { // populates probs
  //   long n_cells = CL_idxs.C_idxs.size();
  //   auto C_indices = torch::from_blob(CL_idxs.C_idxs.data(), {n_cells}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kLong);
  //   auto L_indices = torch::from_blob(CL_idxs.L_idxs.data(), {n_cells}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kLong);
  //   auto indices = torch::stack({C_indices, L_indices})// .to(CUDA_FLAG ? torch::kCUDA : torch :: kCPU)
  //     ;

  //   auto values = torch::ones({n_cells}).to(torch::kFloat32)// .to(CUDA_FLAG ? torch::kCUDA : torch :: kCPU)
  //     ;
  //   int64_t n_clauses = CL_idxs.n_clauses;
  //   int64_t n_lits = CL_idxs.n_vars * 2;
  //   std::vector<int64_t> sizes = {n_clauses, n_lits};
 
  //   auto G = at::sparse_coo_tensor(indices, values, sizes);
  //   if (CUDA_FLAG)
  //     {
  //     G = G.to(torch::kCUDA);
  //   }
  //   std::vector<torch::jit::IValue> inputs = {G};
    
  //   // auto outputs = module.forward(inputs).toTuple();
  //   // auto V_drat_logits = outputs -> elements()[0].toTensor().squeeze();
  //   auto outputs = module.forward(inputs).toTensor();
  //   return outputs.squeeze();

  //   // V_drat_logits = V_drat_logits.view({1, V_drat_logits.size(0)});
  //   // auto V_core_logits = outputs[1];
  // }
}
