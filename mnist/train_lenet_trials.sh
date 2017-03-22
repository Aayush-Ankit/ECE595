#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=ECE595/mnist/solver/lenet_solver0.prototxt $@ 2>&1 | tee solver0_trace.txt 
./build/tools/caffe train --solver=ECE595/mnist/solver/lenet_solver1.prototxt $@ 2>&1 | tee solver1_trace.txt 
./build/tools/caffe train --solver=ECE595/mnist/solver/lenet_solver2.prototxt $@ 2>&1 | tee solver2_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/lenet_solver3.prototxt $@ 2>&1 | tee solver3_trace.txt
