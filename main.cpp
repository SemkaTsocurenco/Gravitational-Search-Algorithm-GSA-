#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <memory>    
#include <random>  
#include <iomanip>  

#define DIMENSIONS 10
#define N_PARTICLES 100
#define MIN_RAND -10
#define MAX_RAND 10



double random(double lo, double hi) {
    static std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}

double target_function (std::vector<double> possitions) {
    double sum;
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



int main() {

    std::vector<std::vector<double>> positions (N_PARTICLES, std::vector<double>(DIMENSIONS));
    std::vector<std::vector<double>> velocities (N_PARTICLES, std::vector<double>(DIMENSIONS, 0.0));

    for (int i = 0; i < N_PARTICLES; ++i)
        for (int d = 0; d < DIMENSIONS; ++d)
            positions[i][d] = random(MIN_RAND, MAX_RAND);

    for (int i = 0; i < N_PARTICLES; ++i) {
        std::cout << std::fixed << std::setprecision(3)<< std::showpos;
        std::cout << "\33[34mParticle " << i << ":\33[0m\n";
        std::cout << "  pos: " << positions[i] << "\n";
        std::cout << "  vel: " << velocities[i] << "\n\n";
    }





    std::unique_ptr<double> global_best_ptr;
    double global_best_fitness2 = std::numeric_limits<double>::infinity();

    return 0;
}