#ifndef INIT_H
#define INIT_H



#include <algorithm>
#include <cmath>
#include <execution>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <Eigen/Dense>                

#ifdef _OPENMP
  #include <omp.h>
#endif



namespace gsa {

// -----------------------------
// 1. Параметры задачи
// -----------------------------
constexpr int    DIMENSIONS   = 1000;
constexpr int    N_PARTICLES  = 300;
constexpr int    ITERATIONS   = 800;
constexpr double MIN_RAND     = -100.0;
constexpr double MAX_RAND     =  100.0;

constexpr double G0           = 100.0;   // начальная гравитац. константа
constexpr double ALPHA        = 20.0;    // коэффициент затухания   G(t)
constexpr double EPS          = 1e-9;    // защита от деления на 0
constexpr double V_MAX        = (MAX_RAND - MIN_RAND) * 0.1;   // «физический» лимит скорости

constexpr double V_MAX_FRAC  = 0.1;    // V_max = frac*(X_MAX-X_MIN)
constexpr int    K_BEST      = N_PARTICLES / 10;   // 10 % лучших

using Vec    = std::vector<double>;
using Matrix = std::vector<Vec>;

// -----------------------------
// 2. Вспомогательные функции
// -----------------------------
double rand_double(double lo, double hi) {
    thread_local static std::mt19937_64 gen{std::random_device{}()};
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(gen);
}

double target(const Vec& x) {                 // f(x) = Σ x_i²
    return std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
}

double dist_sq(const Vec& a, const Vec& b) {  // без √ — быстрее
    double s = 0.0;
    for (int d = 0; d < DIMENSIONS; ++d)
        s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}


std::ostream& operator<<(std::ostream& os, const Vec& v) {
    os << "[ ";
    for (std::size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    os << " ]";
    return os;
}

} // namespace gsa
#endif // INIT_H