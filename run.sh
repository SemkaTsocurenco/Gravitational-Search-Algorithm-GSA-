#!/bin/bash
mkdir build 
mkdir results
cd build 
cmake ../
make -j${nproc}
./gsa_optimised 
cd ../
python3 ./graph.py
