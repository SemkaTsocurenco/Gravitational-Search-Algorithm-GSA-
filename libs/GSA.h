// gsa.hpp  ------------------------------------------------------------------
#pragma once

#include "init.h"             // runtime-параметры + set_objective()/target()
#include "fitness_logger.h"   // лог запусков

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>
#include <vector>

namespace gsa {

/// Одно выполнение классического GSA.
/// @param objective_name  – "sphere", "rosenbrock", "rastrigin" или "ackley"
/// @param run_csv_path    – CSV, куда пишутся агрегированные итоги запуска
inline void GSA(std::string_view objective_name,
                std::string_view run_csv_path = "../results/GSA_runs.csv")
{
    /* ---------- 0. выбор целевой функции ---------- */
    if (!gsa::set_objective(objective_name))
        throw std::runtime_error("GSA: unknown objective '" +
                                 std::string(objective_name) + "'");

    /* ---------- 1. агрегированный лог ---------- */
    RunLogger rlog(run_csv_path,
                   std::string(objective_name),
                   gsa::DIMENSIONS,
                   gsa::N_PARTICLES);

    const auto t_start = std::chrono::high_resolution_clock::now();

    /* ---------- 2. allocate & init ---------- */
    std::vector<gsa::Vec> pos(N_PARTICLES, Vec(DIMENSIONS));
    std::vector<gsa::Vec> vel(N_PARTICLES, Vec(DIMENSIONS, 0.0));
    std::vector<gsa::Vec> acc(N_PARTICLES, Vec(DIMENSIONS, 0.0));

    #pragma omp parallel for
    for (int i = 0; i < N_PARTICLES; ++i)
        for (int d = 0; d < DIMENSIONS; ++d)
            pos[i][d] = rand_double(MIN_RAND, MAX_RAND);

    std::vector<double> fitness(N_PARTICLES);
    std::vector<double> mass(N_PARTICLES);

    double global_best = std::numeric_limits<double>::infinity();
    int stagnant = 0;

    int it = 0;
    for (; it < ITERATIONS; ++it)
    {
        /* 3. fitness */
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            fitness[i] = gsa::target(pos[i]);

        int    best_idx  = std::min_element(fitness.begin(), fitness.end()) - fitness.begin();
        double best_fit  = fitness[best_idx];
        double worst_fit = *std::max_element(fitness.begin(), fitness.end());

        if (best_fit + EPS < global_best) {
            global_best = best_fit;
            stagnant    = 0;
        } else ++stagnant;

        /* 4. mass */
        double denom = worst_fit - best_fit + EPS;
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            mass[i] = (worst_fit - fitness[i]) / denom;
        double inv_sum = 1.0 / (std::accumulate(mass.begin(), mass.end(), 0.0) + EPS);
        for (double &m : mass) m *= inv_sum;

        /* 5. G(t) */
        double G = G0 * std::exp(-ALPHA * it / double(ITERATIONS));

        /* 6. forces */
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_PARTICLES; ++i) {
            std::fill(acc[i].begin(), acc[i].end(), 0.0);
            for (int j = 0; j < N_PARTICLES; ++j) if (i != j) {
                double coef = G * mass[j] / (dist_sq(pos[i], pos[j]) + EPS);
                double rnd  = rand_double(0.0, 1.0);
                for (int d = 0; d < DIMENSIONS; ++d)
                    acc[i][d] += rnd * coef * (pos[j][d] - pos[i][d]);
            }
        }

        /* 7. update */
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            for (int d = 0; d < DIMENSIONS; ++d) {
                double r = rand_double(0.0, 1.0);
                vel[i][d]  = r * vel[i][d] + acc[i][d] / (mass[i] + EPS);
                vel[i][d]  = std::clamp(vel[i][d], -V_MAX(), V_MAX());
                pos[i][d] += vel[i][d];
                pos[i][d]  = std::clamp(pos[i][d], MIN_RAND, MAX_RAND);
            }

        /* 8. early stop */
        if (stagnant >= 30000 || best_fit < 1e-5) break;
    }

    /* ---------- 9. лог итогов ---------- */
    double total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t_start).count();
    rlog.log_result(it + 1, total_ms, global_best);

    std::cout << "\n[GSA | " << objective_name << "]  best=" << global_best
              << "  iter=" << it + 1
              << "  run#" << rlog.run_idx() << '\n';
}

} // namespace gsa