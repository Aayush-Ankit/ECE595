#!/usr/bin/env sh 
set -e
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl1.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl1_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl2.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl2_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl3.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl3_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl1.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl1_trace2.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl2.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl2_trace2.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl3.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl3_trace2.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl1.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl1_trace3.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl2.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl2_trace3.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcl/lenet_solver_fcl3.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcl/solver_fcl3_trace3.txt
