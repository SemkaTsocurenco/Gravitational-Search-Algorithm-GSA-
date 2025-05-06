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
        ofs_.open(csv_path_);
        ofs_.clear();
        if(!ofs_) throw std::runtime_error("Cannot open csv for writing");

        
        if (ofs_.is_open()) ofs_ << "iter,time,bf\n";
    }

    ~FitnessLogger(){
        if(ofs_) ofs_.close();
    }

    void push(int iter, int seconds , double mggsa_bf)
    {
        std::scoped_lock lk(mut_);
        ofs_ << std::setprecision(15 )<< iter << "," << seconds << "," << mggsa_bf << '\n';
    }

private:
    std::string   csv_path_;
    std::ofstream ofs_;
    std::mutex    mut_;
};
