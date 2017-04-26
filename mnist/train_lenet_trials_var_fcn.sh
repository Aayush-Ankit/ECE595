#!/usr/bin/env sh 
set -e
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn100.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn100_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn300.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn300_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn500.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn500_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn100.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn100_trace2.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn300.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn300_trace2.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn500.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn500_trace2.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn100.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn100_trace3.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn300.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn300_trace3.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fcn/lenet_solver_fcn500.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fcn/solver_fcn500_trace3.txt
