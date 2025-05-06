#include "./init.h"
#include "fitness_logger.h"
#include <cmath>

using namespace gsa;                // init.h: constants + rand_double + target()



void GSA()
{
    FitnessLogger flog("../results/GSA_DATA.csv");

    auto t0 = std::chrono::high_resolution_clock::now();

    // ----------------------- allocate & init --------------------------------------
    std::vector<std::vector<double>> pos(N_PARTICLES, std::vector<double>(DIMENSIONS));
    std::vector<std::vector<double>> vel(N_PARTICLES, std::vector<double>(DIMENSIONS, 0.0));
    std::vector<std::vector<double>> acc(N_PARTICLES, std::vector<double>(DIMENSIONS, 0.0));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_PARTICLES; ++i)
        for (int d = 0; d < DIMENSIONS; ++d)
            pos[i][d] = rand_double(MIN_RAND, MAX_RAND);

    std::vector<double> fitness(N_PARTICLES);
    std::vector<double> mass(N_PARTICLES);

    double global_best_fit = std::numeric_limits<double>::infinity();
    std::vector<double> global_best(DIMENSIONS, 0.0);

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        // ---- 1. fitness ----------------------------------------------------------
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_PARTICLES; ++i)
            fitness[i] = target(pos[i]);

        int best_idx = std::min_element(fitness.begin(), fitness.end()) - fitness.begin();
        double best_fit  = fitness[best_idx];
        double worst_fit = *std::max_element(fitness.begin(), fitness.end());

        if (best_fit < global_best_fit) {
            global_best_fit = best_fit;
            global_best     = pos[best_idx];
        }

        // ---- 2. mass ------------------------------------------------------------
        double denom = worst_fit - best_fit + EPS;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_PARTICLES; ++i)
            mass[i] = (worst_fit - fitness[i]) / denom;          // heavier = better

        double mass_sum = 0.0;
        #pragma omp parallel for reduction(+:mass_sum)
        for (int i = 0; i < N_PARTICLES; ++i) mass_sum += mass[i];
        double inv_sum = 1.0 / (mass_sum + EPS);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_PARTICLES; ++i) mass[i] *= inv_sum;

        // ---- 3. G(t) ------------------------------------------------------------
        double G = G0 * std::exp(-ALPHA * double(iter) / ITERATIONS);

        // ---- 4. forces / accelerations -----------------------------------------
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_PARTICLES; ++i)
        {
            std::fill(acc[i].begin(), acc[i].end(), 0.0);
            for (int j = 0; j < N_PARTICLES; ++j)
            {
                if (i == j) continue;
                double dist_ij = dist_sq(pos[i], pos[j]);
                double coef    = G * mass[j] / dist_ij;
                double rand_val = rand_double(0.0, 1.0);
                for (int d = 0; d < DIMENSIONS; ++d)
                    acc[i][d] += rand_val * coef * (pos[j][d] - pos[i][d]);
            }
        }

        // ---- 5. update velocity & position -------------------------------------
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_PARTICLES; ++i)
            for (int d = 0; d < DIMENSIONS; ++d)
            {
                double r = rand_double(0.0, 1.0);
                vel[i][d]  = r * vel[i][d] + acc[i][d] / (mass[i] + EPS);
                vel[i][d]  = std::clamp(vel[i][d], -V_MAX, V_MAX);
                pos[i][d] += vel[i][d];
                pos[i][d]  = std::clamp(pos[i][d], MIN_RAND, MAX_RAND);
            }


        // ---- 6. log -------------------------------------------------------------
        auto now = std::chrono::high_resolution_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
        flog.push(iter, ms, global_best_fit);

    }
    std::cout << "\nGSA final best=" << global_best_fit << '\n';
}
