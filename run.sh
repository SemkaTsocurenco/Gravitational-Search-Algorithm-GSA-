#!/bin/bash
mkdir build 
mkdir results
cd build 
cmake ../
make -j${nproc}
./GSA
cd ../
python3 ./animation.py
