#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>
#include <memory>      

#define DIMENTIONS 10
#define N_PARTICLES 100
#define MIN_RAND -10
#define MAX_RAND 10


double random(double min, double max)
{
    return (double)(rand())/RAND_MAX*(max - min) + min; 
}


double target_function (std::vector<double> possitions) {
    double sum;
    for (auto p : possitions){
        sum += pow(p,2);
    }
    return sum;
}




int main() {
    std::srand(time(NULL));

    std::vector<std::vector<double>> possiions (N_PARTICLES, std::vector<double>(DIMENTIONS));
    std::vector<std::vector<double>> velocities (N_PARTICLES, std::vector<double>(DIMENTIONS));

    for (int N = 0 ; N < N_PARTICLES ; N++){
        for (int D = 0 ; D < DIMENTIONS ; D++){     
            possiions[N][D] = random(MIN_RAND, MAX_RAND);
            velocities[N][D] = 0.0;
        }
    }



    std::unique_ptr<double> global_best_ptr;
    double global_best_fitness2 = std::numeric_limits<double>::infinity();





    std::cout<<"\nAA\n";



    // for (int i = 30 ;;){
    //     std::cout<<"\r\33["<<std::to_string(i)<<"m Hello World!"<<std::flush;
    //     i = (i > 35) ? (30) : (i + 1);
    // }

    return 0;
}