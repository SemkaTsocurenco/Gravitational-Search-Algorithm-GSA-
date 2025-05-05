#include "./GSA.h"
#include "./MGGSA.h"
#include "fitness_logger.h"

#include <chrono>
#include <ctime>
#include <locale>


int main() {
    FitnessLogger flog;          
    GSA();
    MGGSA();
    return 0;
}
