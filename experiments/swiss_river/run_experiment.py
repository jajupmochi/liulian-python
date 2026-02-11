"""Swiss River forecasting experiment using Time-LLM model.

This script follows the training loop from Time-LLM's run_main.py closely,
adapted to work within the liulian project structure.

Source reference:
  refer_projects/Time-LLM_20260209_154911/run_main.py

Data:
  Swiss River Network dataset (1990 stations)
  refer_projects/Time-LLM_20260209_154911/dataset/swiss_river/

Model:
  TimeLLM (adapted in liulian/models/torch/timellm.py)

Usage:
  # Train (default config):
  python experiments/swiss_river/run_experiment.py

  # Train with custom config:
  python experiments/swiss_river/run_experiment.py --config experiments/swiss_river/configs/swiss_river.yaml

  # Evaluate only:
  python experiments/swiss_river/run_experiment.py --eval_only

  # Quick test (1 epoch, small batch):
  python experiments/swiss_river/run_experiment.py --quick_test
"""

import argparse
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: add Time-LLM reference project for data_provider and utils
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TIMELLM_ROOT = os.path.join(PROJECT_ROOT, 'refer_projects', 'Time-LLM_20260209_154911')

# Add reference project to path for data_provider and utils imports
if TIMELLM_ROOT not in sys.path:
    sys.path.insert(0, TIMELLM_ROOT)

from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate

# Import adapted TimeLLM model from liulian
sys.path.insert(0, PROJECT_ROOT)
from liulian.models.torch.timellm import Model as TimeLLMModel


# ---------------------------------------------------------------------------
# EarlyStopping (from utils/tools.py, fixed for NumPy 2.0)
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.

    Source: refer_projects/Time-LLM_20260209_154911/utils/tools.py
    Fixed: np.Inf → np.inf (NumPy 2.0 compatibility)
    """

    def __init__(self, patience=7, verbose=False, delta=0, save_mode=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → '
                  f'{val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint'))
        self.val_loss_min = val_loss


# ---------------------------------------------------------------------------
# Prompt content loader (adapted for swiss river naming)
# ---------------------------------------------------------------------------
# Prompt bank file mapping (data name → prompt file)
PROMPT_FILE_MAP = {
    'swiss-river-1990': 'wt-swiss-1990',
    'swiss-river-2010': 'wt-swiss-2010',
    'swiss-river-zurich': 'wt-zurich',
}


def load_content(args):
    """Load prompt content for Time-LLM. Handles swiss river file naming."""
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = PROMPT_FILE_MAP.get(args.data, args.data)
    prompt_path = os.path.join(TIMELLM_ROOT, 'dataset', 'prompt_bank', f'{file}.txt')
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            return f.read()
    # Fallback: generic description
    return (
        'Swiss River Network water temperature dataset. '
        'Contains daily water and air temperature measurements from monitoring stations.'
    )


# ---------------------------------------------------------------------------
# Validation function (adapted from utils/tools.py::vali)
# ---------------------------------------------------------------------------
def validate(args, model, vali_loader, criterion, mae_metric, device):
    """Run validation loop and return average MSE and MAE losses."""
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # Decoder input: label_len of ground truth + zeros for pred_len
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, :args.label_len, :], dec_inp], dim=1
            ).float().to(device)

            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            loss = criterion(outputs, batch_y)
            mae_loss = mae_metric(outputs, batch_y)
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    model.train()
    return np.average(total_loss), np.average(total_mae_loss)


# ---------------------------------------------------------------------------
# Training function (from run_main.py inline loop)
# ---------------------------------------------------------------------------
def train(args, device):
    """Full training loop following Time-LLM's run_main.py."""
    print(f'=== Training: {args.model_id} ===')
    print(f'Model: {args.model}, Data: {args.data}')
    print(f'seq_len={args.seq_len}, pred_len={args.pred_len}, features={args.features}')
    print(f'LLM backbone: {args.llm_model} (dim={args.llm_dim}, layers={args.llm_layers})')
    print(f'Device: {device}')

    for ii in range(args.itr):
        # Setting record
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data,
            args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers,
            args.d_ff, args.factor, args.embed, args.des, ii)

        # Data
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        print(f'Train samples: {len(train_data)}, Val samples: {len(vali_data)}, '
              f'Test samples: {len(test_data)}')

        # Model
        model = TimeLLMModel(args).float().to(device)

        # Load prompt content
        args.content = load_content(args)

        # Checkpoint path
        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        os.makedirs(path, exist_ok=True)

        # Collect trainable parameters (LLM backbone is frozen)
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in trained_parameters)
        print(f'Total params: {total_params:,}, Trainable: {trainable_params:,} '
              f'({100 * trainable_params / total_params:.1f}%)')

        # Optimizer
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        # Scheduler
        train_steps = len(train_loader)
        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=args.pct_start,
                epochs=args.train_epochs,
                max_lr=args.learning_rate)

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()
        early_stopping = EarlyStopping(patience=args.patience)

        # Training loop
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
                    enumerate(train_loader), total=len(train_loader),
                    desc=f'Epoch {epoch + 1}/{args.train_epochs}'):

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
                dec_inp = torch.cat(
                    [batch_y[:, :args.label_len, :], dec_inp], dim=1
                ).float().to(device)

                # Forward
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - epoch_time) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} '
                          f'| speed: {speed:.4f}s/iter | ETA: {left_time:.0f}s')

                loss.backward()
                model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(None, model_optim, scheduler, epoch + 1,
                                         args, printout=False)
                    scheduler.step()

            epoch_cost = time.time() - epoch_time
            train_loss_avg = np.average(train_loss)

            vali_loss, vali_mae = validate(
                args, model, vali_loader, criterion, mae_metric, device)
            test_loss, test_mae = validate(
                args, model, test_loader, criterion, mae_metric, device)

            print(f'Epoch {epoch + 1} ({epoch_cost:.1f}s) | '
                  f'Train: {train_loss_avg:.7f} | '
                  f'Vali: {vali_loss:.7f} | '
                  f'Test: {test_loss:.7f} (MAE: {test_mae:.7f})')

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print('Early stopping triggered.')
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                    adjust_learning_rate(None, model_optim, scheduler, epoch + 1,
                                         args, printout=True)

        # Load best model and evaluate
        best_model_path = os.path.join(path, 'checkpoint')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f'Loaded best model from {best_model_path}')

        final_test_loss, final_test_mae = validate(
            args, model, test_loader, criterion, mae_metric, device)
        print(f'\n=== Final Test Results (iteration {ii}) ===')
        print(f'MSE: {final_test_loss:.7f}, MAE: {final_test_mae:.7f}')

    return model


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------
def evaluate(args, device):
    """Load trained model and evaluate on test set."""
    print(f'=== Evaluation: {args.model_id} ===')

    test_data, test_loader = data_provider(args, 'test')
    print(f'Test samples: {len(test_data)}')

    model = TimeLLMModel(args).float().to(device)

    # Find checkpoint
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers,
        args.d_ff, args.factor, args.embed, args.des, 0)
    ckpt_path = os.path.join(args.checkpoints, setting + '-' + args.model_comment, 'checkpoint')

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'Loaded checkpoint: {ckpt_path}')
    else:
        print(f'WARNING: No checkpoint found at {ckpt_path}, using random init.')

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    test_loss, test_mae = validate(args, model, test_loader, criterion, mae_metric, device)
    print(f'Test MSE: {test_loss:.7f}, Test MAE: {test_mae:.7f}')

    return test_loss, test_mae


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description='Swiss River Time-LLM Experiment')

    # Experiment control
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides CLI args)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, evaluate from checkpoint')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test: 1 epoch, batch_size=4')

    # Basic config (same as Time-LLM run_main.py)
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='swiss_river_timellm')
    parser.add_argument('--model_comment', type=str, default='swiss_river_timellm')
    parser.add_argument('--model', type=str, default='TimeLLM')
    parser.add_argument('--seed', type=int, default=2021)

    # Data loader
    parser.add_argument('--data', type=str, default='swiss-river-1990')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Root path for dataset (default: auto-detect)')
    parser.add_argument('--data_path', type=str, default='swiss-1990.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--loader', type=str, default='modal')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Forecasting
    parser.add_argument('--seq_len', type=int, default=90)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=7)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

    # Model architecture
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--prompt_domain', type=int, default=0)
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_dim', type=int, default=768)

    # Optimization
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--des', type=str, default='swiss_river')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)

    return parser


def load_config_yaml(args, config_path):
    """Load YAML config and override args (same logic as run_main.py debug mode)."""
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    print(f'Loaded config from {config_path}')
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load YAML config if provided
    if args.config is not None:
        args = load_config_yaml(args, args.config)
    else:
        # Default: load from experiments/swiss_river/configs/swiss_river.yaml
        default_config = os.path.join(SCRIPT_DIR, 'configs', 'swiss_river.yaml')
        if os.path.exists(default_config):
            args = load_config_yaml(args, default_config)

    # Auto-detect root_path if not set
    if args.root_path is None or not os.path.isabs(args.root_path):
        # Use the data from Time-LLM reference project
        candidate = os.path.join(TIMELLM_ROOT, 'dataset', 'swiss_river')
        if os.path.isdir(candidate):
            args.root_path = candidate + '/'
            print(f'Auto-detected data root: {args.root_path}')
        else:
            args.root_path = os.path.join(TIMELLM_ROOT,
                                          args.root_path or 'dataset/swiss_river/')

    # Quick test overrides
    if args.quick_test:
        args.train_epochs = 1
        args.batch_size = 4
        args.num_workers = 0
        args.patience = 1
        print('Running in quick_test mode (1 epoch, batch_size=4)')

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make checkpoints path absolute
    if not os.path.isabs(args.checkpoints):
        args.checkpoints = os.path.join(PROJECT_ROOT, args.checkpoints)

    # Set CWD to Time-LLM root so data_provider paths resolve correctly
    original_cwd = os.getcwd()
    os.chdir(TIMELLM_ROOT)

    try:
        if args.eval_only:
            evaluate(args, device)
        else:
            train(args, device)
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    main()
