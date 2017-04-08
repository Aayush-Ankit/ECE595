#!/usr/bin/env sh 
set -e
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl1_fcn100.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl1_fcn100_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl1_fcn300.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl1_fcn300_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl1_fcn500.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl1_fcn500_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl2_fcn100.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl2_fcn100_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl2_fcn300.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl2_fcn300_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl2_fcn500.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl2_fcn500_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl3_fcn100.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl3_fcn100_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl3_fcn300.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl3_fcn300_trace.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_fc_size/lenet_solver_fcl3_fcn500.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_fc_size/solver_fcl3_fcn500_trace.txt
