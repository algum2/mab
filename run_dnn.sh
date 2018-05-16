#!/bin/bash
cd "$(dirname "$0")"

source ~/.bashrc

function linear()
{
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --epsilon=0.0  > DNN_linear_eg_0.0
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --epsilon=1.0  > DNN_linear_eg_1.0

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --epsilon=0.05 > DNN_linear_eg_0.05 
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --epsilon=0.1  > DNN_linear_eg_0.1
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --epsilon=0.15 > DNN_linear_eg_0.15

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --keep_prob=0.8  > DNN_linear_dropout_0.8
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --keep_prob=0.85 > DNN_linear_dropout_0.85
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --keep_prob=0.9  > DNN_linear_dropout_0.9
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --keep_prob=0.95 > DNN_linear_dropout_0.95

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --model_num=2 > DNN_linear_bootstrap_2
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --model_num=5 > DNN_linear_bootstrap_5

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --input_dummy=True --dummy_lr=0.001 > DNN_linear_input_0.001
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --input_dummy=True --dummy_lr=0.005 > DNN_linear_input_0.005
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --input_dummy=True --dummy_lr=0.01  > DNN_linear_input_0.01

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --action_dummy=True --dummy_lr=0.001 > DNN_linear_action_0.001
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --action_dummy=True --dummy_lr=0.005 > DNN_linear_action_0.005
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --action_dummy=True --dummy_lr=0.01  > DNN_linear_action_0.01
}

function nonlinear
{
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --epsilon=0.0  > DNN_nonlinear_eg_0.0
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --epsilon=1.0  > DNN_nonlinear_eg_1.0

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --epsilon=0.05 > DNN_nonlinear_eg_0.05 
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --epsilon=0.1  > DNN_nonlinear_eg_0.1
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --epsilon=0.15 > DNN_nonlinear_eg_0.15

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --no-linear --keep_prob=0.8  > DNN_nonlinear_dropout_0.8
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --no-linear --keep_prob=0.85 > DNN_nonlinear_dropout_0.85
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --no-linear --keep_prob=0.9  > DNN_nonlinear_dropout_0.9
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --no-linear --keep_prob=0.95 > DNN_nonlinear_dropout_0.95

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --no-linear --model_num=2 > DNN_nonlinear_bootstrap_2
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --no-linear --model_num=5 > DNN_nonlinear_bootstrap_5

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --input_dummy=True --dummy_lr=0.001 > DNN_nonlinear_input_0.001
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --input_dummy=True --dummy_lr=0.005 > DNN_nonlinear_input_0.005
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --input_dummy=True --dummy_lr=0.01  > DNN_nonlinear_input_0.01

    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --action_dummy=True --dummy_lr=0.001 > DNN_nonlinear_action_0.001
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --action_dummy=True --dummy_lr=0.005 > DNN_nonlinear_action_0.005
    CUDA_VISIBLE_DEVICES=8 python ./run_dnn.py --no-linear --action_dummy=True --dummy_lr=0.01  > DNN_nonlinear_action_0.01
}

linear & 
nonlinear &

