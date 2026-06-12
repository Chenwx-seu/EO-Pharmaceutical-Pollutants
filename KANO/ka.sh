python train.py \
    --data_path LOK_train_22.csv \
    --features_path train_22.npz \
    --no_features_scaling \
    --separate_test_path LOK_test_22.csv \
    --separate_test_features_path test_22.npz \
    --dataset_type regression \
    --exp_name KANO_regression \
    --epochs 130\
    --dropout 0.030054205722373103 \
    --batch_size 4 \
    --init_lr 0.0011925487545389197 \
    --depth 3 \
    --ffn_num_layers 3 \
    --exp_id ka \
    --step functional_prompt \
    --hidden_size 407 \
    --seed 42 \

