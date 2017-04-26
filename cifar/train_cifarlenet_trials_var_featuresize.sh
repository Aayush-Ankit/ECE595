#!/usr/bin/env sh 
set -e
./build/tools/caffe train --solver=ECE595/cifar/solver/var_num_feature_maxiter6000_lr0.01/cifarlenet_solver_featuresize10.prototxt $@ 2>&1 | tee ECE595/cifar/traces/var_num_feature_maxiter6000_lr0.01/solver_featuresize10_trace.txt
./build/tools/caffe train --solver=ECE595/cifar/solver/var_num_feature_maxiter6000_lr0.01/cifarlenet_solver_featuresize25.prototxt $@ 2>&1 | tee ECE595/cifar/traces/var_num_feature_maxiter6000_lr0.01/solver_featuresize25_trace.txt
./build/tools/caffe train --solver=ECE595/cifar/solver/var_num_feature_maxiter6000_lr0.01/cifarlenet_solver_featuresize50.prototxt $@ 2>&1 | tee ECE595/cifar/traces/var_num_feature_maxiter6000_lr0.01/solver_featuresize50_trace.txt
./build/tools/caffe train --solver=ECE595/cifar/solver/var_num_feature_maxiter6000_lr0.01/cifarlenet_solver_featuresize100.prototxt $@ 2>&1 | tee ECE595/cifar/traces/var_num_feature_maxiter6000_lr0.01/solver_featuresize100_trace.txt
./build/tools/caffe train --solver=ECE595/cifar/solver/var_num_feature_maxiter6000_lr0.01/cifarlenet_solver_featuresize150.prototxt $@ 2>&1 | tee ECE595/cifar/traces/var_num_feature_maxiter6000_lr0.01/solver_featuresize150_trace.txt
