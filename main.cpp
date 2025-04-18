#include <iostream>
#include <thread>
#include <chrono>

int main() {

    for (int i = 30 ;;){
        std::cout<<"\r\33["<<std::to_string(i)<<"m Hello World!"<<std::flush;
        i = (i > 35) ? (30) : (i + 1);
    }

    return 0;
}