#include "internal.hpp"

namespace CaDiCaL {

  bool Internal::refocusing () {
    return opts.refocus && stats.conflicts > lim.query;
  }
  
// This initializes variables on the binary 'scores' heap also with
// smallest variable index first (thus picked first) and larger indices at
// the end.
//
void Internal::init_scores (int old_max_var, int new_max_var) {
  LOG ("initializing EVSIDS scores from %d to %d",
    old_max_var + 1, new_max_var);
  for (int i = old_max_var + 1; i <= new_max_var; i++)
    scores.push_back (i);
}

// Refocus the EVSIDS heap.
void Internal::refocus_scores () {
  refocused = true;
      printf("REFOCUSING SCORES\n");  
  // auto CL_idxs = CLIndices();
  // auto nv_to_v = std::vector<int>();
  auto [CL_idxs, nv_to_v] = buildCLIndices();
  lim.query = stats.conflicts + opts.queryinterval;

  torch::Tensor V_logits;
  if (!opts.randomrefocus)
    {
      
  
      // ############# DEBUGGING DUMP ######################################
      // try
      //   {  
      //     auto [CL_idxs, nv_to_v] = buildCLIndices();
      //     FILE* f;
      //     if (dump_dir_set_flag)
      //       {
      //         char dump_path[255];
      //         std::sprintf(dump_path, "%srefocus_dump_%lu.cnf", dump_dir, refocus_dump_count);
      //         f = fopen(dump_path, "wb");
      //       }
      //     else f = stdout;

      //     CL_idxs.dump(f);
      //     refocus_dump_count++;      
      //   }
      // catch (std::runtime_error &e)
      //   {
      //     std::cout << e.what() << "\n";
      //     return  ;
      //   }
      // ###################################################################  
  
      V_logits = gnn1(CL_idxs);

    }
  else
    {
      V_logits = torch::rand(CL_idxs.n_vars).to(torch::kFloat32);
    }

  auto V_probs = torch::softmax(V_logits * 4.0, 0);
  for (unsigned v_idx = 0; v_idx < nv_to_v.size(); v_idx++)
    {
      auto idx = nv_to_v[v_idx] + 1;
      score (idx) = opts.refocusscale * nv_to_v.size() * V_probs[v_idx].item<double>();
      if (scores.contains (idx))
        {
          scores.update (idx);
        }
    }
};

// Shuffle the EVSIDS heap.

void Internal::shuffle_scores () {
  if (!opts.shuffle) return;
  if (!opts.shufflescores) return;
  assert (!level);
  stats.shuffled++;
  LOG ("shuffling scores");
  vector<int> shuffle;
  if (opts.shufflerandom){
    scores.erase ();
    for (int idx = max_var; idx; idx--)
      shuffle.push_back (idx);
    Random random (opts.seed);                  // global seed
    random += stats.shuffled;                   // different every time
    for (int i = 0; i <= max_var-2; i++) {
      const int j = random.pick_int (i, max_var-1);
      swap (shuffle[i], shuffle[j]);
    }
  } else {
    while (!scores.empty ()) {
      int idx = scores.front ();
      (void) scores.pop_front ();
      shuffle.push_back (idx);
    }
  }
  scinc = 0;
  for (const auto & idx : shuffle) {
    stab[idx] = scinc++;
    scores.push_back (idx);
  }
}

}
