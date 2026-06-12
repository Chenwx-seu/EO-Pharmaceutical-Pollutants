#!/bin/bash
# chemprop_pipeline.sh
# Usage: bash chemprop_pipeline.sh

set -e  


TRAIN_CSV="LOK_train_155.csv"
TEST_CSV="LOK_test_155.csv"
TRAIN_FEAT="train_155.csv"
TEST_FEAT="test_155.csv"
HYPEROPT_DIR="data355"
CONFIG_SAVE="args.json"
BEST_CONFIG="best_hy.json"
MODEL_DIR="best_model"


echo "========== Step 1: Hyperparameter Search =========="
chemprop_hyperopt \
    --data_path $TRAIN_CSV \
    --separate_test_path $TEST_CSV \
    --features_path $TRAIN_FEAT \
    --separate_test_features_path $TEST_FEAT \
    --no_features_scaling \
    --dataset_type regression \
    --num_iters 200 \
    --search_parameter_keywords all \
    --num_workers 14 \
    --hyperopt_checkpoint_dir $HYPEROPT_DIR \
    --config_save_path $CONFIG_SAVE


echo "========== Step 2: Training =========="
chemprop_train \
    --data_path $TRAIN_CSV \
    --separate_test_path $TEST_CSV \
    --features_path $TRAIN_FEAT \
    --separate_test_features_path $TEST_FEAT \
    --no_features_scaling \
    --dataset_type regression \
    --epochs 120 \
    --config_path $BEST_CONFIG \
    --save_dir $MODEL_DIR


echo "========== Step 3: Predict Train Set =========="
chemprop_predict \
    --test_path $TRAIN_CSV \
    --features_path $TRAIN_FEAT \
    --checkpoint_path $MODEL_DIR/fold_0/model_0/model.pt \
    --no_features_scaling \
    --preds_path predictions_train.csv


echo "========== Step 4: Predict Test Set =========="
chemprop_predict \
    --test_path $TEST_CSV \
    --features_path $TEST_FEAT \
    --checkpoint_path $MODEL_DIR/fold_0/model_0/model.pt \
    --no_features_scaling \
    --preds_path predictions_test.csv

echo "========== All Done =========="