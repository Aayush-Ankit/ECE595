#!/usr/bin/env sh 
set -e
./build/tools/caffe train --solver=ECE595/mnist/solver/var_num_feature_maxiter100/lenet_solver_featuresize10.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_num_feature_maxiter100/solver_featuresize10_trace3.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_num_feature_maxiter100/lenet_solver_featuresize25.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_num_feature_maxiter100/solver_featuresize25_trace3.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_num_feature_maxiter100/lenet_solver_featuresize50.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_num_feature_maxiter100/solver_featuresize50_trace3.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_num_feature_maxiter100/lenet_solver_featuresize100.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_num_feature_maxiter100/solver_featuresize100_trace3.txt
./build/tools/caffe train --solver=ECE595/mnist/solver/var_num_feature_maxiter100/lenet_solver_featuresize150.prototxt $@ 2>&1 | tee ECE595/mnist/traces/var_num_feature_maxiter100/solver_featuresize150_trace3.txt
