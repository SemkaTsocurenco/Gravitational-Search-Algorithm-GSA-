// mggsa.hpp  ---------------------------------------------------------------
#pragma once
#include "init.h"
#include "fitness_logger.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>
#include <vector>

namespace gsa {

/* ---------- Быстрый MDS (N×D → N×2) ------------- */
inline Eigen::MatrixXd mds2D(const Matrix& pos)
{
    Eigen::MatrixXd B(N_PARTICLES, N_PARTICLES);   //   больше не static
    #pragma omp parallel for
    for (int i = 0; i < N_PARTICLES; ++i) {
        B(i,i) = 0.0;
        for (int j = i + 1; j < N_PARTICLES; ++j) {
            double d = std::sqrt(dist_sq(pos[i], pos[j]));
            B(i,j) = B(j,i) = d;
        }
    }

    Eigen::VectorXd rowMean = B.rowwise().mean();
    double totalMean        = rowMean.mean();

    Eigen::MatrixXd C = -0.5 * (B.array().square().matrix());
    C.colwise()      -= rowMean;
    C.rowwise()      -= rowMean.transpose();
    C.array()        += totalMean;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    Eigen::VectorXd vals = es.eigenvalues().tail(2)
                               .cwiseMax(0.0).cwiseSqrt();
    return es.eigenvectors().rightCols(2) * vals.asDiagonal();
}

/* ---------- λ‑коэффициенты ---------------------- */
inline Vec lambda_coeff(const Matrix& pos,
                        const Vec&    fitness)
{
    Eigen::MatrixXd Y = mds2D(pos);                //   локальная переменная

    /* 1. индексы K_best */
    const int K = K_BEST();
    std::vector<int> idx(N_PARTICLES);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b){ return fitness[a] < fitness[b]; });
    idx.resize(K);

    /* 2. ранги */
    auto to_rank = [](const Vec& v){
        std::vector<int> id(v.size()); std::iota(id.begin(), id.end(), 0);
        std::sort(id.begin(), id.end(),
                  [&](int a, int b){ return v[a] < v[b]; });
        std::vector<int> r(v.size());
        for (int p = 0; p < (int)v.size(); ++p) r[id[p]] = p;
        return r;
    };

    Vec s_high(K), s_low(K);
    for (int t = 0; t < K; ++t) {
        s_high[t] = fitness[idx[t]];
        s_low [t] = Y.row(idx[t]).norm();
    }
    auto r_high = to_rank(s_high);
    auto r_low  = to_rank(s_low);

    /* 3. diff[d] */
    Vec diff(DIMENSIONS, 0.0);
    for (int t = 0; t < K; ++t) {
        double dlt = std::abs(r_high[t] - r_low[t]);
        for (int d = 0; d < DIMENSIONS; ++d) diff[d] += dlt;
    }
    double diff_max = *std::max_element(diff.begin(), diff.end()) + EPS;
    for (auto& v : diff) v /= diff_max;

    /* 4. группировка и λ */
    Vec lambda(DIMENSIONS, 1.0);
    for (int g = 0; g < L_GROUPS; ++g) {
        double low  = double(g) / L_GROUPS;
        double high = double(g + 1) / L_GROUPS;
        std::vector<int> dimlist;
        for (int d = 0; d < DIMENSIONS; ++d)
            if (diff[d] >= low && diff[d] < high) dimlist.push_back(d);

        if (dimlist.empty()) continue;

        double delta = 0.0;
        for (int d : dimlist) delta += diff[d];
        delta /= dimlist.size();

        double lambda_g = 1.0 / (delta + 0.01);
        for (int d : dimlist) lambda[d] = lambda_g;
    }
    double mean = std::accumulate(lambda.begin(), lambda.end(), 0.0) /
                  DIMENSIONS + EPS;
    for (auto& l : lambda) l /= mean;
    return lambda;
}

/* ---------- Основная функция ------------------------------------------- */
inline void MGGSA(std::string_view objective,
                  std::string_view run_csv = "../results/MGGSA_runs.csv")
{
    if (!gsa::set_objective(objective))
        throw std::runtime_error("MGGSA: bad objective");

    RunLogger rlog(run_csv,
                   std::string(objective),
                   DIMENSIONS,
                   N_PARTICLES);

    auto t0 = std::chrono::high_resolution_clock::now();

    /* allocate */
    Matrix pos(N_PARTICLES, Vec(DIMENSIONS));
    Matrix vel(N_PARTICLES, Vec(DIMENSIONS, 0.0));
    Matrix acc(N_PARTICLES, Vec(DIMENSIONS, 0.0));
    #pragma omp parallel for
    for (int i = 0; i < N_PARTICLES; ++i)
        for (int d = 0; d < DIMENSIONS; ++d)
            pos[i][d] = rand_double(MIN_RAND, MAX_RAND);

    Vec fitness(N_PARTICLES), mass(N_PARTICLES);
    Vec lambda(DIMENSIONS, 1.0);

    Vec best_pos(DIMENSIONS, 0.0);
    double best_fit = std::numeric_limits<double>::infinity();
    int stagn = 0, it = 0;

    for (; it < ITERATIONS; ++it)
    {
        /* fitness */
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            fitness[i] = target(pos[i]);

        int best_idx = std::min_element(fitness.begin(), fitness.end()) - fitness.begin();
        double f_best = fitness[best_idx],
               f_worst = *std::max_element(fitness.begin(), fitness.end());

        if (f_best + EPS < best_fit) { best_fit = f_best; stagn = 0; }
        else ++stagn;

        /* masses */
        double denom = f_worst - f_best + EPS;
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            mass[i] = (f_worst - fitness[i]) / denom;
        double inv_sum = 1.0 /
            (std::accumulate(mass.begin(), mass.end(), 0.0) + EPS);
        for (double &m : mass) m *= inv_sum;

        /* λ каждые MDSTEP */
        if (it % MDSTEP == 0) lambda = lambda_coeff(pos, fitness);

        /* forces */
        for (auto &a : acc) std::fill(a.begin(), a.end(), 0.0);
        double G = G0 * std::exp(-ALPHA * it / double(ITERATIONS));

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_PARTICLES; ++i)
            for (int j = 0; j < N_PARTICLES; ++j) if (i != j) {
                double dist = std::sqrt(dist_sq(pos[i], pos[j])) + EPS;
                double coef = G * mass[j] / dist * rand_double(0.0, 1.0);
                for (int d = 0; d < DIMENSIONS; ++d)
                    acc[i][d] += coef * lambda[d] * (pos[j][d] - pos[i][d]);
            }

        /* update */
        #pragma omp parallel for
        for (int i = 0; i < N_PARTICLES; ++i)
            for (int d = 0; d < DIMENSIONS; ++d) {
                vel[i][d] = rand_double(0.0, 1.0) * vel[i][d] + acc[i][d];
                vel[i][d] = std::clamp(vel[i][d], -V_MAX(), V_MAX());
                pos[i][d] += vel[i][d];
                pos[i][d]  = std::clamp(pos[i][d], MIN_RAND, MAX_RAND);
            }

        if (stagn >= 30000 || best_fit < 1e-5) break;
    }

    double total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0).count();
    rlog.log_result(it + 1, total_ms, best_fit);

    std::cout << "\n[MGGSA | " << objective << "]  best = "
              << best_fit << "   iterations = " << it + 1
              << "   run #" << rlog.run_idx() << '\n';
}

} // namespace gsa
