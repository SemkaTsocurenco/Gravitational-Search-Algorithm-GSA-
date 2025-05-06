// init.h -------------------------------------------------------------------
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
#include <string_view>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace gsa {

// -----------------------------
// 1. Параметры задачи (runtime-modifiable)
// -----------------------------
inline int    DIMENSIONS   = 2;
inline int    N_PARTICLES  = 400;
inline int    ITERATIONS   = 2000;
inline double MIN_RAND     = -100.0;
inline double MAX_RAND     =  100.0;

inline double G0           = 100.0;   // начальная G
inline double ALPHA        = 10.0;    // затухание
inline double EPS          = 1e-9;    // защита от деления на 0
inline double V_MAX_FRAC   = 0.1;     // V_max = frac*(X_MAX−X_MIN)
inline int    L_GROUPS     = 5;       // число групп в MGGSA
inline int    MDSTEP       = 10;      // период пересчёта λ
inline int    K_MIN        = 5;       // нижняя граница k-best

// вычисляем на лету, чтобы учесть возможное изменение N_PARTICLES и X_MIN/MAX
inline double V_MAX()    { return (MAX_RAND - MIN_RAND) * V_MAX_FRAC; }
inline int    K_BEST()   { return std::max(K_MIN, N_PARTICLES / 10); }

using Vec    = std::vector<double>;
using Matrix = std::vector<Vec>;

// -----------------------------
// 2. Случайные числа
// -----------------------------
inline double rand_double(double lo, double hi)
{
    thread_local static std::mt19937_64 gen{std::random_device{}()};
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(gen);
}


// -----------------------------
// 3. Определения целевых функций
// -----------------------------
enum class Objective { Sphere, Rosenbrock, Rastrigin, Ackley };

/// f(x) = Σ xᵢ²
inline double sphere(const Vec& x)
{
    return std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
}

/// f(x) = Σ[100(xᵢ₊₁ − xᵢ²)² + (xᵢ − 1)²]
inline double rosenbrock(const Vec& x)
{
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i)
        s += 100.0 * std::pow(x[i + 1] - x[i] * x[i], 2)
           + std::pow(x[i] - 1.0, 2);
    return s;
}

/// f(x) = 10n + Σ[xᵢ² − 10 cos(2πxᵢ)]
inline double rastrigin(const Vec& x)
{
    constexpr double A = 10.0;
    constexpr double TAU = 2.0 * 3.14159265358979323846;
    double s = A * x.size();
    for (double xi : x)
        s += xi * xi - A * std::cos(TAU * xi);
    return s;
}

/// f(x) = −20 exp(−0.2√(1/n Σxᵢ²)) − exp(1/n Σcos(2πxᵢ)) + 20 + e
inline double ackley(const Vec& x)
{
    constexpr double PI  = 3.14159265358979323846;
    constexpr double A   = 20.0;
    constexpr double B   = 0.2;
    constexpr double C   = 2.0 * PI;

    const double n = static_cast<double>(x.size());
    double sum_sq = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    double sum_cos = 0.0;
    for (double xi : x) sum_cos += std::cos(C * xi);

    return -A * std::exp(-B * std::sqrt(sum_sq / n))
           - std::exp(sum_cos / n)
           + A + std::exp(1.0);
}

// -----------------------------
// 4. Выбор и вызов целевой функции
// -----------------------------
using TargetFn = double(*)(const Vec&);
inline TargetFn   TARGET = sphere;          ///< указатель на активную функцию
inline Objective  CURRENT_OBJECTIVE = Objective::Sphere;

/// Установить новую целевую функцию
inline void set_objective(Objective obj)
{
    CURRENT_OBJECTIVE = obj;
    switch (obj) {
        case Objective::Sphere:      TARGET = sphere;      break;
        case Objective::Rosenbrock:  TARGET = rosenbrock;  break;
        case Objective::Rastrigin:   TARGET = rastrigin;   break;
        case Objective::Ackley:      TARGET = ackley;      break;
    }
}

/// Установить по строке, например set_objective("rastrigin")
inline bool set_objective(std::string_view name)
{
    if      (name == "sphere")      set_objective(Objective::Sphere);
    else if (name == "rosenbrock")  set_objective(Objective::Rosenbrock);
    else if (name == "rastrigin")   set_objective(Objective::Rastrigin);
    else if (name == "ackley")      set_objective(Objective::Ackley);
    else return false;
    return true;
}

/// Универсальный вызов из алгоритма оптимизации
inline double target(const Vec& x) { return TARGET(x); }

// -----------------------------
// 4. Вспомогательные функции
// -----------------------------
inline double dist_sq(const Vec& a, const Vec& b)
{
    double s = 0.0;
    for (int d = 0; d < DIMENSIONS; ++d)
        s += (a[d] - b[d]) * (a[d] - b[d]);
    return s;
}

inline std::ostream& operator<<(std::ostream& os, const Vec& v)
{
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
