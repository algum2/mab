#!/bin/bash
cd "$(dirname "$0")"

source ~/.bashrc

function linear()
{
    python run_lr.py --alpha=0 > LR_linear_ucb_0
    python run_lr.py --alpha=1 > LR_linear_ucb_1

    python run_lr.py --alpha=0.05 > LR_linear_ucb_0.05
    python run_lr.py --alpha=0.10 > LR_linear_ucb_0.1
    python run_lr.py --alpha=0.15 > LR_linear_ucb_0.15

    python run_lr.py --dummy_lr=0.001 > LR_linear_dummy_0.001
    python run_lr.py --dummy_lr=0.005 > LR_linear_dummy_0.005
    python run_lr.py --dummy_lr=0.01  > LR_linear_dummy_0.01
}

function nonlinear()
{
    python run_lr.py --no-linear --alpha=0.00 > LR_nonlinear_ucb_0.0
    python run_lr.py --no-linear --alpha=0.05 > LR_nonlinear_ucb_0.05
    python run_lr.py --no-linear --alpha=0.10 > LR_nonlinear_ucb_0.1

    python run_lr.py --no-linear --dummy_lr=0.001 > LR_nonlinear_dummy_0.001
    python run_lr.py --no-linear --dummy_lr=0.005 > LR_nonlinear_dummy_0.005
    python run_lr.py --no-linear --dummy_lr=0.01  > LR_nonlinear_dummy_0.01
}

linear &
nonlinear &

