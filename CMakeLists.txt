cmake_minimum_required(VERSION 3.18)
project(gsa_optimised LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include_directories("./libs/")
add_executable(gsa_optimised main.cpp)

find_package(Eigen3 REQUIRED)
target_link_libraries(gsa_optimised PRIVATE Eigen3::Eigen)

# ---- OpenMP (опционально) ----------------------------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found – enabling")
    target_compile_definitions(gsa_optimised PRIVATE _OPENMP)
    target_link_libraries(gsa_optimised PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- oneTBB (нужен для std::execution::par) ---------------------------------
find_package(TBB REQUIRED COMPONENTS tbb)
target_link_libraries(gsa_optimised PRIVATE TBB::tbb)

# ---- Доп. оптимизации (не MSVC) ---------------------------------------------
if (NOT MSVC)
    target_compile_options(gsa_optimised PRIVATE -O3 -march=native -Wall -Wextra)
endif()
