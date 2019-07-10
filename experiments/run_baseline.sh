#!/bin/bash

if [ "$#" -eq 1 -a "$1" = "w" ]
then
    python baseline.py --model_type full --kernel_type CylindricalKernel --model_update 1 --problem Rosenbrock --num_dims 20 --acq_rule EI --bayesian --n_init 2 --seed 117 -w
else
    python baseline.py --model_type full --kernel_type CylindricalKernel --model_update 1 --problem Rosenbrock --num_dims 20 --acq_rule EI --bayesian --n_init 2 --seed 117
fi