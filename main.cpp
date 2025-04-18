#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <memory>    
#include <random>  
#include <iomanip>  
#include <algorithm> 
#include <iterator>  

// -----------------------------
// 1. Задание параметров задачи
// -----------------------------
#define DIMENSIONS 2
#define N_PARTICLES 300
#define ITERATION 40
#define MIN_RAND -10
#define MAX_RAND 10

#define G0 100       // Начальное значение гравитационной константы
#define alpha 20     // Параметр экспоненциального затухания G(t)
#define eps 1e-6     // Маленькая константа для предотвращения деления на ноль


double random(double lo, double hi) {
    static std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}



double target_function (std::vector<double> possitions) {
    double sum = 0;
    for (auto p : possitions){
        sum += pow(p,2);
    }
    return sum;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v){
    os<< "[ ";
    for (int i= 0 ; i < v.size(); i++){
        os<<"\33[32m"<<v[i]<<"\33[0m";
        if (i + 1 < v.size()) 
            os << ", ";
    }
    os<< " ]";
    return os;
}

double calculate_dist(std::vector<double> pos_i, std::vector<double> pos_j){
    double sum = 0;
    for (int d = 0; d<DIMENSIONS; d++){
        sum += pow((pos_i[d]-pos_j[d]),2);
    }
    return pow (sum, 0.5);
}


int main() {

    std::vector<std::vector<double>> positions (N_PARTICLES, std::vector<double>(DIMENSIONS));
    std::vector<std::vector<double>> velocities (N_PARTICLES, std::vector<double>(DIMENSIONS, 0.0));
    std::vector<std::vector<double>> forces (N_PARTICLES, std::vector<double>(DIMENSIONS, 0.0));

    for (int i = 0; i < N_PARTICLES; ++i)
        for (int d = 0; d < DIMENSIONS; ++d)
            positions[i][d] = random(MIN_RAND, MAX_RAND);

    for (int i = 0; i < N_PARTICLES; ++i) {
        std::cout << std::fixed << std::setprecision(3)<< std::showpos;
        std::cout << "\33[34mParticle " << i << ":\33[0m\n";
        std::cout << "  pos: " << positions[i] << "\n";
        std::cout << "  vel: " << velocities[i] << "\n\n";
    }


    std::unique_ptr<std::vector<double>> global_best_ptr;
    double global_best_fitness = std::numeric_limits<double>::infinity();
    std::vector<double> fitness(N_PARTICLES);
    std::vector<double> masses;
    std::vector<double> history;

    double G;

    for (int iter = 0; iter < ITERATION; ++iter) {
        for (int i = 0; i < N_PARTICLES; ++i)
            fitness[i] = target_function(positions[i]);

        auto min_it = std::min_element(fitness.begin(), fitness.end());
        size_t min_index = std::distance(fitness.begin(), min_it);

        // Обновляем глобальное лучшее решение
        if (*min_it < global_best_fitness) {
            global_best_fitness = *min_it;
            global_best_ptr = std::make_unique<std::vector<double>>(positions[min_index]);
        }

        history.push_back(global_best_fitness);

        // Найдем best и worst фитнес
        double best_fit = *std::min_element(fitness.begin(), fitness.end());
        double worst_fit = *std::max_element(fitness.begin(), fitness.end());

        // -------------------------------------------
        // 3. Преобразование фитнеса в массы
        // -------------------------------------------
        masses.clear();
        masses.reserve(N_PARTICLES);
        if (best_fit == worst_fit) {
            for (int i = 0; i < N_PARTICLES; ++i) {
                masses.push_back(1.0);
            }
        } else {
            for (int i = 0; i < N_PARTICLES; ++i) {
                masses.push_back((fitness[i] - worst_fit) / (best_fit - worst_fit));
            }
        }

        // 4. Нормировка масс
        double mass_sum = std::accumulate(masses.begin(), masses.end(), 0.0);
        for (auto& m : masses) {
            m /= (mass_sum + eps);
        }
        // ------------------------------------------------------
        // 4. Вычисление гравитационной константы G(t)
        // ------------------------------------------------------
        G = G0 * std::exp(- alpha * static_cast<double>(iter) / ITERATION);
        
        // ----------------------------------------------------
        // 5. Расчёт сил между частицами
        // ----------------------------------------------------
        for (auto& f : forces)
            std::fill(f.begin(), f.end(), 0.0);
        for (int i = 0; i<N_PARTICLES; i++){
            for (int j = 0; j<N_PARTICLES; j++){
                if (i == j) continue;
                double distance;
                distance = calculate_dist(positions[i], positions[j]);
                for (int d = 0 ; d<DIMENSIONS; d++){
                    double rand_val = random(0.0, 1.0);
                    forces[i][d] += (rand_val * G * masses[i] * masses[j] * (positions[j][d] - positions[i][d]) / (distance + eps));
                }

            }
        }
        // ---------------------------------------------
        // 6. Обновление ускорения, скорости и позиций
        // ---------------------------------------------

        for (int N = 0; N<N_PARTICLES; N++){
            for (int D = 0 ; D<DIMENSIONS; D++){
                double rand_val = random(0.0, 1.0);
                velocities[N][D] = rand_val * velocities[N][D] + (forces[N][D] / (masses[N] + eps));
                positions[N][D] += velocities[N][D];
                if (positions[N][D] < MIN_RAND)
                    positions[N][D] = MIN_RAND;
                if (positions[N][D] > MAX_RAND)
                    positions[N][D] = MAX_RAND;

            }
        }

    }

    std::cout << "\33[31mGlobal best fitness:\33[0m "
            << global_best_fitness << "\n"
            << "\33[31mGlobal best position:\33[0m "
            << *global_best_ptr << "\n";

    for (int i = 0; i < history.size(); ++i)
        std::cout << "iter=" << i
            << "  best_fitness=" << history[i] << "\n";
    


    return 0;
}