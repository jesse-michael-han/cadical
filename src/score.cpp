#include "internal.hpp"

namespace CaDiCaL {

double Internal::glr ()  {
  return ((double) ((double) stats.conflicts / (double) (stats.decisions + 1)));
}

bool Internal::refocusing () {
  if (stable)
    {
  return opts.refocus &&
         // stats.stabconflicts > lim.stabquery &&
    (opts.refocusreluctant ? refocus_reluctant : (stats.conflicts > lim.query)) &&
    refocused &&
    (opts.refocusgluesucks ? glue_sucks(opts.refocusgluesucksmargin) : true) &&
    (process_time() > opts.refocusinittime);
    }
  else
    {
      return opts.refocus &&
        // stats.unstabconflicts > lim.unstabquery &&
        (opts.refocusreluctant ? refocus_reluctant : (stats.conflicts > lim.query)) &&
        // refocused &&
        (opts.refocusgluesucks ? glue_sucks(opts.refocusgluesucksmargin) : true) &&
        (process_time() > opts.refocusinittime);
      // return false;
    }

    // if (stats.conflicts > 50e6) return false;
    // return opts.refocus && stats.conflicts > lim.query && (!(level < refocus_skip_level));
  // return opts.refocus &&
  //        stats.conflicts > lim.query &&
  //        refocused &&
  //        (opts.refocusgluesucks ? (use_scores()) || glue_sucks(opts.refocusgluesucksmargin) : true) &&
         // (process_time() > opts.refocusinittime);
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
  auto start = process_time();
  refocused = false;

  // if (stable)
  //   {
  //     auto next_interval = min((double) opts.refocusceil, opts.queryinterval + (opts.refocusdecaybase * (pow((double) stats.stab_refocus_count + 1, opts.refocusdecayexp))));
  // lim.stabquery = stats.stabconflicts + next_interval;
  // std::cout << "NEXT STABLE INTERVAL " << next_interval << "\n";

  //   }
  // else
  //   {
  //     auto next_interval = min((double) opts.refocusceil, opts.queryinterval + (opts.refocusdecaybase * (pow((double) stats.unstab_refocus_count + 1, (opts.refocusdecayexp * 1.5)))));
  // lim.unstabquery = stats.unstabconflicts + next_interval;
  // std::cout << "NEXT UNSTABLE INTERVAL " << next_interval << "\n";
  //   }

  if (opts.refocusreluctant)
    {
      std::cout << "NEXT INTERVAL " << refocus_reluctant.get_countdown() << "\n";
    }
  else
    {
      auto next_interval = min((double) opts.refocusceil, opts.queryinterval + (opts.refocusdecaybase * (pow((double) stats.refocus_count + 1, opts.refocusdecayexp))));
      lim.query = stats.conflicts + next_interval;

      std::cout << "NEXT INTERVAL " << next_interval << "\n";
    }


  printf("REFOCUSING SCORES\n");

  auto pr = buildCLIndices();
  auto CL_idxs = std::get<0>(pr);
  auto nv_to_v = std::get<1>(pr);

  if (nv_to_v.size() == 0)
    {
      printf("TOO BIG\n");
      stats.oom_count++;
      // auto elapsed = process_time() - start;

      // stats.total_refocus_time += elapsed;
      // refocus_skip_level = level;

      return;
    }

  // lim.query = stats.conflicts + min(opts.queryinterval + ((pow((double) stats.refocus_count + 1, 1.45)) * 10e3), 150e3);

  // auto V_logits = torch::empty(nv_to_v.size(), gnn1.CUDA_FLAG ? torch::kCUDA : torch::kCPU );

  // if (!opts.randomrefocus)
  //   {


  //     // ############# DEBUGGING DUMP ######################################
  //     // try
  //     //   {
  //     //     auto [CL_idxs, nv_to_v] = buildCLIndices();
  //     //     FILE* f;
  //     //     if (dump_dir_set_flag)
  //     //       {
  //     //         char dump_path[255];
  //     //         std::sprintf(dump_path, "%srefocus_dump_%lu.cnf", dump_dir, refocus_dump_count);
  //     //         f = fopen(dump_path, "wb");
  //     //       }
  //     //     else f = stdout;

  //     //     CL_idxs.dump(f);
  //     //     refocus_dump_count++;
  //     //   }
  //     // catch (std::runtime_error &e)
  //     //   {
  //     //     std::cout << e.what() << "\n";
  //     //     return  ;
  //     //   }
  //     // ###################################################################

  //     V_logits = gnn1(CL_idxs);

  //   }
  // else
  //   {
  //     V_logits = torch::rand(CL_idxs.n_vars).to(torch::kFloat32);
  //   }
  try
    {
      torch::Tensor V_logits;
      if (opts.randomrefocus) {
        V_logits = gnn1(CL_idxs);
        V_logits = torch::rand(CL_idxs.n_vars).to(torch::kFloat32);        
      }
      else {
        V_logits = gnn1(CL_idxs);
      }

      // auto V_logits = (!opts.randomrefocus) ? gnn1(CL_idxs) : torch::rand(CL_idxs.n_vars).to(torch::

      auto update_scores = [&]()
                         {
                           auto V_probs = torch::softmax(V_logits * 4.0, 0);
                           // V_probs *= (1.0 - pow(((double) ((double) stats.conflicts / (double) (stats.decisions + 1))), 2.0));

                           for (unsigned v_idx = 0; v_idx < nv_to_v.size(); v_idx++)
                             {
                               auto idx = nv_to_v[v_idx] + 1;
                               // score (idx) = opts.refocusscale * nv_to_v.size() * V_probs[v_idx].item<double>();
                               // score (idx) += scinc * opts.refocusscale * V_probs[v_idx].item<double>() * (1.0 - pow(((double) ((double) stats.conflicts / (double) (stats.decisions + 1))), 2.0)); 
                               if (opts.refocusrebump) score (idx) += scinc * opts.refocusscale * V_probs[v_idx].item<double>() * (0.25 + 0.75 * glr());
                               else score (idx) = opts.refocusscale * nv_to_v.size() * V_probs[v_idx].item<double>();
                               if (scores.contains (idx))
                                 {
                                   scores.update (idx);
                                 }
                             }
                           // scinc = 1.0; // reset score increment to 1 if we're refocusing and not rebumping
                         };

      if (use_scores())
        {
          // if (opts.randomrefocus) V_logits = torch::rand(CL_idxs.n_vars).to(torch::kFloat32);
          update_scores();
        }
      else // in unstable phase, so reorder the queue
        {
          // if (opts.randomrefocus) V_logits = torch::rand(CL_idxs.n_vars).to(torch::kFloat32);
          // update_scores();
          std::vector<std::pair<int, double>> updates;
          // auto V_probs = torch::softmax(V_logits * 4.0, 0);
          // V_logits = torch::rand(CL_idxs.n_vars).to(torch::kFloat32);
          auto V_logits_size = V_logits.size(0);
          double BUMP_FRAC = 0.75; // bump top (1-BUMP_FRAC) variables as scored by the network
          std::vector<int> to_bump(V_logits_size);
          std::iota (std::begin(to_bump), std::end(to_bump), 0);
          std::sort(to_bump.begin(), to_bump.end(), [&](int x, int y) { return (V_logits[x]< V_logits[y]).item<bool>();});
           for (auto it = to_bump.begin() + floor(BUMP_FRAC * V_logits_size); it < to_bump.end(); it++) {
            bump_queue(nv_to_v[*it]+1);
          }
        }

      if (stable) { stats.stab_refocus_count++; }
      else { stats.unstab_refocus_count++; }

      stats.refocus_count++;

      auto elapsed = process_time() - start;

      stats.total_refocus_time += elapsed;
      // stats.avg_refocus_time = ((stats.refocus_count - 1) * stats.avg_refocus_time + elapsed)/(stats.refocus_count);
      return;

    }
  catch (std::runtime_error& e) // in case CUDA OOM error when running on GPU
    {
      std::cout << "CAUGHT RUNTIME ERROR\n"  << e.what() << "\n";
      stats.oom_count++;
      return;
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
