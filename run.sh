#!/bin/bash
mkdir build 
cd build 
cmake ../
make -j${nproc}
./GSA
cd ../
python3 ./animation.py
