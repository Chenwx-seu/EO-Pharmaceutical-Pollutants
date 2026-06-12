import argparse
import subprocess
import re
import optuna
import numpy as np

def parse_rmse_from_output(output_lines):

    best_rmse = float('inf')
    for line in output_lines:
        match = re.search(r"Validation rmse =\s*([0-9.]+)", line)
        if match:
            best_rmse = min(best_rmse, float(match.group(1)))
    return best_rmse

def objective(trial, args):

    init_lr = trial.suggest_float('init_lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    depth = trial.suggest_int('depth', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    hidden_size = trial.suggest_int('hidden_size', 300, 1000)
    ffn_num_layers = trial.suggest_int('ffn_num_layers', 1, 4)

    exp_name = f"trial_lr{init_lr}_bs{batch_size}_hs{hidden_size}_d{depth}_do{dropout:.2f}_ffn{ffn_num_layers}"


    cmd = [
        "python", "-u", "train.py",
        "--dataset_type", "regression",
        "--data_path", args.data_path,
        "--features_path", args.features_path,
        "--separate_test_path", args.separate_test_path,
        "--separate_test_features_path", args.separate_test_features_path,
        "--no_features_scaling",
        "--epochs", str(args.epochs),
        "--batch_size", str(batch_size),
        "--hidden_size", str(hidden_size),
        "--depth", str(depth),
        "--dropout", str(dropout),
        "--init_lr", str(init_lr),
        "--ffn_num_layers", str(ffn_num_layers),
        "--exp_name", exp_name,
        "--exp_id", "optuna_trial",
        "--step", "functional_prompt"
    ]

    print("\n[Trial] Running command:", " ".join(cmd))


    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line.strip())
    process.wait()

    rmse = parse_rmse_from_output(output_lines)
    print(f"[Trial] RMSE: {rmse:.4f}")

    return rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='LOK_train_22.csv')
    parser.add_argument('--features_path', type=str, default='train_22.npz')
    parser.add_argument('--separate_test_path', type=str, default='LOK_test_22.csv')
    parser.add_argument('--separate_test_features_path', type=str, default='test_22.npz')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("----activation", type=str, default='ReLU')
    args = parser.parse_args()


    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, n_jobs=args.n_jobs)

  
    print("\n========== Best Trial ==========")
    trial = study.best_trial
    print(f"  Best RMSE: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
