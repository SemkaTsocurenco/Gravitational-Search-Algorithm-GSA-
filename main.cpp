// main.cpp
#include "init.h"
#include "GSA.h"
#include "MGGSA.h"
#include <iostream>
#include <string>

using namespace gsa;  // чтобы не писать каждый раз gsa::

int main() {
    // фиксированное число итераций
    ITERATIONS = 90000;

    // список целевых функций
    const char* objectives[] = {"sphere", "rosenbrock", "rastrigin", "ackley"};
    const std::string run_csv = "runs.csv";

    for (int N = 50; N <= 200; N += 50) {
        for (int D = 20; D <= 100; D += 10) {
            N_PARTICLES = N;
            DIMENSIONS  = D;

            for (auto obj : objectives) {
                std::cout << "\n=== N=" << N
                          << "  D=" << D
                          << "  OBJ=" << obj << " ===\n";

                std::string trace_gsa = "trace_gsa_" + std::to_string(N)
                                      + "x" + std::to_string(D)
                                      + "_" + obj + ".csv";
                std::string trace_mgg = "trace_mggsa_" + std::to_string(N)
                                      + "x" + std::to_string(D)
                                      + "_" + obj + ".csv";

                GSA(obj);
                MGGSA(obj);
            }
        }
    }
    return 0;
}
