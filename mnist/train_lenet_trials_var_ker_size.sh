#!/usr/bin/env sh 
set -e
./build/tools/caffe train --solver=ECE595/mnist/solver/var_ker_size/lenet_solver_kersize3.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_ker_size/solver_kersize3_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_ker_size/lenet_solver_kersize5.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_ker_size/solver_kersize5_trace.txt
