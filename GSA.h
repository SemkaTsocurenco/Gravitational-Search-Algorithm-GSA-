#include "./init.h"

void GSA(){
  using namespace gsa;

    // --- 3. Инициализация ----------------------------------------------------
    Matrix pos(N_PARTICLES, Vec(DIMENSIONS));
    Matrix vel(N_PARTICLES, Vec(DIMENSIONS, 0.0));
    Matrix acc(N_PARTICLES, Vec(DIMENSIONS, 0.0));

    std::for_each(std::execution::par_unseq, pos.begin(), pos.end(),
        [](Vec& p) {
            std::generate(p.begin(), p.end(),
                [] { return rand_double(MIN_RAND, MAX_RAND); });
        });

    Vec fitness(N_PARTICLES);
    Vec masses (N_PARTICLES);

    Vec    global_best(DIMENSIONS, 0.0);
    double global_best_fit = std::numeric_limits<double>::infinity();

    std::vector<double> history;
    history.reserve(ITERATIONS);

    std::ofstream best_csv("best_history.csv");
    best_csv << "iteration,best_fitness\n";

    // --- 4. Главный цикл -----------------------------------------------------
    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        // 4.1 fitness ---------------------------------------------------------
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            fitness[i] = target(pos[i]);

        auto [min_it, max_it] = std::minmax_element(fitness.begin(), fitness.end());
        double best_fit  = *min_it;
        double worst_fit = *max_it;
        std::size_t best_idx = std::distance(fitness.begin(), min_it);

        if (best_fit < global_best_fit) {
            global_best_fit = best_fit;
            global_best     = pos[best_idx];
        }

        history.push_back(global_best_fit);
        best_csv << iter << ',' << global_best_fit << '\n';

        // 4.2 массы -----------------------------------------------------------
        double denom = worst_fit - best_fit + EPS;
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            masses[i] = (worst_fit - fitness[i]) / denom;

        double mass_sum = std::reduce(std::execution::par_unseq,
                                      masses.begin(), masses.end(), 0.0);
        for (auto& m : masses) m /= (mass_sum + EPS);

        // 4.3 g(t) и k‑best ----------------------------------------------------
        double G = G0 * std::exp(-ALPHA * static_cast<double>(iter) / ITERATIONS);

        int K = std::max(1,
                 static_cast<int>(N_PARTICLES *
                 (1.0 - static_cast<double>(iter) / ITERATIONS)));

        std::vector<int> idx(N_PARTICLES);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&fitness](int a, int b) { return fitness[a] < fitness[b]; });

        // 4.4 силы и ускорения -------------------------------------------------
        for (auto& a : acc) std::fill(a.begin(), a.end(), 0.0);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_PARTICLES; ++i) {
            for (int k = 0; k < K; ++k) {
                int j = idx[k];
                if (i == j) continue;

                double d2   = dist_sq(pos[i], pos[j]);
                double dist = std::sqrt(d2) + EPS;
                double coef = G * masses[j] / dist;            // ← m_j / r

                for (int d = 0; d < DIMENSIONS; ++d)
                    acc[i][d] += rand_double(0.0, 1.0) * coef *
                                 (pos[j][d] - pos[i][d]);
            }
        }

        // 4.5 скорость и позиция ----------------------------------------------
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            for (int d = 0; d < DIMENSIONS; ++d) {
                vel[i][d]  = rand_double(0.0, 1.0) * vel[i][d] + acc[i][d];
                vel[i][d]  = std::clamp(vel[i][d], -V_MAX, V_MAX);

                pos[i][d] += vel[i][d];
                pos[i][d]  = std::clamp(pos[i][d], MIN_RAND, MAX_RAND);
            }
    }

    // --- 5. Итог -------------------------------------------------------------
    std::cout << "\nGlobal best fitness  : " << global_best_fit  << '\n'
              <<   "Global best position : " << global_best      << '\n';

}