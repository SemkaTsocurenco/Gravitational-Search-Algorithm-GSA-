// fitness_logger.hpp -------------------------------------------------------
#pragma once
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <mutex>

class FitnessLogger {
public:
    explicit FitnessLogger(const std::string& path = "fitness_trace.csv")
        : csv_path_(path)
    {
        bool exists = std::filesystem::exists(csv_path_);
        ofs_.open(csv_path_, std::ios::app);
        if(!ofs_) throw std::runtime_error("Cannot open csv for writing");

        if(!exists) ofs_ << "gsa_bf,mggsa_bf\n";
    }

    void push(double gsa_bf, double mggsa_bf)
    {
        std::scoped_lock lk(mut_);
        ofs_ << std::setprecision(15) << gsa_bf << ',' << mggsa_bf << '\n';
    }

private:
    std::string   csv_path_;
    std::ofstream ofs_;
    std::mutex    mut_;
};
