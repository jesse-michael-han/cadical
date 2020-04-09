#include "internal.hpp"

namespace CaDiCaL
{
  auto readFile(FILE* f, int mlen = 0) {
    assert(f);
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buffer = (char *) malloc(length + 1 + mlen);
    buffer[length] = '\0';
    fread(buffer, 1, length, f);
    fclose(f);
    return std::pair(buffer, length+1);
}

bool Internal::dumping ()  {
  if (opts.dump)
    {
      if (stats.conflicts > lim.dump)
        {
          lim.dump = stats.conflicts + opts.dumpfreq;
          return true;
        }
    }
  return false;
}

// Only useful for debugging purposes.

void Internal::dump () {
    int n_vars = max_var+1;

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

  int64_t m = assumptions.size ();
  int64_t m_irr = assumptions.size();
  // for (int idx = 1; idx <= max_var; idx++)
  //   if (fixed (idx)) {m++; m_irr++;};
  for (const auto & c : clauses)
    if (!c->garbage)
      {
        m++;
        if (!c->redundant)
          {
            m_irr++;
          }
      };

  int64_t CLAUSE_LIMIT = 5e6;

  int64_t REDUNDANT_LIMIT = opts.dumplim * (m_irr + 1);

  if (m_irr > CLAUSE_LIMIT) return;
  FILE * dump_file = stdout;
  if (dump_dir_set_flag)
  {
    char dump_path[255];
    std::sprintf(dump_path, "%sdump_%lu.cnf", dump_dir, dump_count);

    dump_file = fopen(dump_path, "wb");
  }

  int new_max_var = nv_to_v.size();
  // for (int idx = 1; idx <= max_var; idx++) {
  //   const int tmp = fixed (idx);
  //   if (tmp) fprintf (dump_file, "%d 0\n", tmp < 0 ? -idx : idx);
  // }
  int64_t push_count = 0;

  auto dump_clause = [&](Clause * c, FILE * out) // precondition (inefficient because i'm lazy): clause is not satisfied
  {
  // if (c -> redundant)
    for (const auto & lit : *c)
      {
        auto v_idx = vidx(lit) - 1;
        // auto foo = v_to_nv[v_idx] + 1;
        // auto l_abs = externalize(foo);
        auto l_abs = v_to_nv[v_idx] + 1;
        auto new_l = sign(lit) * l_abs;

        if (l_abs == 0) continue;
        
        fprintf (out, "%d ", new_l);
      }

    fprintf (out, "0");
    // if (c -> redundant) {
    //   fprintf (out, " L");
    // }
    fprintf (out, "\n");
    push_count++;
  };

  auto tmp_dump_file = tmpfile();

  auto tmp_dump_file2 = tmpfile();

  for (const auto & c : clauses)
    {
      if (push_count > REDUNDANT_LIMIT) {std::cout << " REDUNDANT LIMIT REACHED \n"; break;};
      for (auto &l : trail) {        
        for (const auto &l2 : *c)
          if (sign(l) == sign(l2)) {if (vidx(l) == vidx(l2)) {goto case_sat;}}
      }

      if (!c->garbage) dump_clause (c, tmp_dump_file);
    case_sat: {};
    }

  auto [body, mlen] = readFile(tmp_dump_file);

  // char header [mlen+255];

  // for (auto &l : trail) {
  //   fprintf(tmp_dump_file2, "%d ", l);
  // }
  // fprintf(tmp_dump_file2, "\n");

  fprintf (tmp_dump_file2, "p cnf %d %" PRId64 "\n", new_max_var, push_count);

  // sprintf(header, "RAMALAMADINGDONG\n");

  auto [header, _] = readFile(tmp_dump_file2, mlen);

  auto foo = strcat(header, body);

  fprintf(dump_file, "%s", foo);

  // for (const auto & lit : assumptions)
  //   printf ("%d 0\n", lit);
  fflush (dump_file);
  if (dump_dir_set_flag) fclose(dump_file);
  dump_count++;
  }


}
