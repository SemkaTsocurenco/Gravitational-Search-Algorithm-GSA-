// run_logger.hpp  -----------------------------------------------------------
#pragma once
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

/*
 * CSV‑схема (одна строка на ЗАПУСК):
 *
 * Objective,D,N,RunIdx,Iterations,Elapsed_ms,BestFitness
 * Sphere,30,50,1,4000,128.7,1.3e‑7
 * Sphere,30,50,2,4000,129.1,9.9e‑8
 * Rastrigin,100,200,1,8000,512.4,3.4e‑1
 *
 * ── Пояснение столбцов ────────────────────────────────────────────────────
 * Objective     – имя целевой функции
 * D             – размерность задачи
 * N             – количество агентов
 * RunIdx        – порядковый № запуска при данной (Objective,D,N),
 *                 нумерация с 1
 * Iterations    – всего выполнено итераций
 * Elapsed_ms    – полное время работы алгоритма (double, мс)
 * BestFitness   – лучший найденный fitness
 */

class RunLogger {
public:
    /// @param csv_path   – путь к общему trace‑файлу
    /// @param objective  – имя целевой функции
    /// @param D          – размерность задачи
    /// @param N          – размер популяции
    explicit RunLogger(std::string_view csv_path,
                       std::string_view objective,
                       int              D,
                       int              N)
        : csv_path_{csv_path}
        , objective_{objective}
        , D_{D}
        , N_{N}
        , run_idx_{determine_next_run_idx()}
    {
        // если файл ещё не существует – запишем заголовок
        if (!std::filesystem::exists(csv_path_)) {
            std::ofstream ofs(csv_path_);
            if (!ofs) throw std::runtime_error(
                "RunLogger: cannot create CSV '" +
                std::string(csv_path_) + '\'');
            ofs << "Objective,D,N,RunIdx,Iterations,Elapsed_ms,BestFitness\n";
        }
    }

    /// Сохраняет агрегированные результаты текущего запуска
    void log_result(int    iterations,
                    double elapsed_ms,
                    double best_fitness)
    {
        std::scoped_lock lk(mut_);
        std::ofstream ofs(csv_path_, std::ios::app);
        if (!ofs) throw std::runtime_error(
            "RunLogger: cannot open CSV '" +
            std::string(csv_path_) + "' for appending");

        ofs << objective_       << ','
            << D_               << ','
            << N_               << ','
            << run_idx_         << ','
            << iterations       << ','
            << std::setprecision(15)
            << elapsed_ms       << ','
            << best_fitness     << '\n';
    }

    /// Текущий порядковый номер запуска (может пригодиться вызывающей стороне)
    [[nodiscard]] int run_idx() const noexcept { return run_idx_; }

private:
    // Определяем следующий RunIdx, подсчитав строки с той же (obj,D,N)
    int determine_next_run_idx() const
    {
        if (!std::filesystem::exists(csv_path_))
            return 1;

        std::ifstream ifs(csv_path_);
        if (!ifs) return 1;

        std::string line;
        int count = 0;
        // пропускаем заголовок
        std::getline(ifs, line);

        const std::string key = objective_ + ',' +
                                std::to_string(D_) + ',' +
                                std::to_string(N_) + ',';
        while (std::getline(ifs, line)) {
            if (line.rfind(key, 0) == 0) // строка начинается с key
                ++count;
        }
        return count + 1;
    }

    const std::string  csv_path_;
    const std::string  objective_;
    const int          D_;
    const int          N_;
    const int          run_idx_;
    mutable std::mutex mut_;
};
