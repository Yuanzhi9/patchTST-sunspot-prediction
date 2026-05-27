import argparse
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.exp_main import Exp_Main
from data_provider.phase_loader import phase_data_provider
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import time


PHASE_LABELS = ['rise', 'peak', 'decline', 'trough']
MODULES_TO_SKIP = {'exp', 'data_provider'}


class PhaseTrainer(Exp_Main):
    def __init__(self, args, phase_name):
        self.phase_name = phase_name
        super().__init__(args)

    def _get_data(self, flag):
        return phase_data_provider(self.args, flag, self.phase_name)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        best_val_loss = float('inf')

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                    dim=1).float().to(self.device)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1,
                                        self.args, printout=False)
                    scheduler.step()

            val_loss = self.vali(vali_data, vali_loader, criterion)
            train_loss = np.average(train_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                }, os.path.join(path, 'checkpoint.pth'))

            early_stopping(val_loss, self.model, model_optim, scheduler,
                          epoch, self.args, path)
            if early_stopping.early_stop:
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

            if epoch % 2 == 0:
                print(f"  [{self.phase_name}] Epoch {epoch+1}/{self.args.train_epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        ckpt_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])

        return best_val_loss

    def evaluate(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        preds, trues = [], []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                    dim=1).float().to(self.device)

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if preds.ndim >= 3:
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        return mae, mse, rmse, preds, trues


def build_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    args.is_training = 1
    args.model_id = 'sunspot_phase'
    args.model = 'PatchTST'
    args.data = 'custom'

    args.features = 'M'
    args.target = 'ssn'
    args.freq = 'm'

    args.seq_len = 24
    args.label_len = 12
    args.pred_len = 12
    args.enc_in = 3
    args.dec_in = 3
    args.c_out = 1

    args.d_model = 128
    args.n_heads = 16
    args.e_layers = 3
    args.d_layers = 1
    args.d_ff = 256
    args.factor = 1

    args.patch_len = 12
    args.stride = 6
    args.padding_patch = 'end'
    args.revin = 1
    args.affine = 0
    args.subtract_last = 0
    args.decomposition = 0
    args.kernel_size = 25
    args.individual = 0

    args.dropout = 0.05
    args.fc_dropout = 0.05
    args.head_dropout = 0.0
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.output_attention = False

    args.train_epochs = 20
    args.batch_size = 16
    args.learning_rate = 0.0001
    args.patience = 8
    args.loss = 'mse'
    args.lradj = 'type3'
    args.pct_start = 0.3

    args.root_path = './dataset_phase/'
    args.checkpoints = './checkpoints_phase/'
    args.num_workers = 0
    args.itr = 1
    args.des = 'phase_test'

    args.use_gpu = False
    args.use_multi_gpu = False
    args.use_amp = False
    args.do_predict = False
    args.test_flop = False
    args.distil = True
    args.random_seed = 2021

    return args


def main():
    args = build_args()
    random = __import__('random')
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    os.makedirs('output', exist_ok=True)

    print("=" * 60)
    print("  分阶段 PatchTST 训练")
    print(f"  seq_len={args.seq_len} pred_len={args.pred_len} d_model={args.d_model}")
    print(f"  batch_size={args.batch_size} epochs={args.train_epochs}")
    print("=" * 60)

    results = {}

    for phase in PHASE_LABELS:
        print(f"\n{'='*40}")
        print(f"  训练阶段: {phase.upper()}")
        print(f"{'='*40}")

        trainer = PhaseTrainer(args, phase)
        setting = f"sunspot_{phase}_{args.seq_len}_{args.pred_len}_PatchTST"
        val_loss = trainer.train(setting)
        mae, mse, rmse, preds, trues = trainer.evaluate(setting)

        results[phase] = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'val_loss': float(val_loss),
            'n_test_samples': len(preds),
        }
        print(f"  [{phase}] MAE={mae:.3f}  MSE={mse:.3f}  RMSE={rmse:.3f}")
        print(f"  [{phase}] 测试样本数={len(preds)}")

    print(f"\n{'='*60}")
    print("  分阶段训练总结果")
    print(f"{'='*60}")
    for phase in PHASE_LABELS:
        r = results[phase]
        print(f"  {phase:8s}  MAE={r['mae']:6.2f}  RMSE={r['rmse']:6.2f}  "
              f"val_loss={r['val_loss']:.4f}  samples={r['n_test_samples']}")

    with open('output/phase_results.txt', 'w') as f:
        f.write("Phase Training Results\n")
        f.write(f"seq_len={args.seq_len} pred_len={args.pred_len} d_model={args.d_model}\n\n")
        for phase in PHASE_LABELS:
            r = results[phase]
            f.write(f"{phase}: MAE={r['mae']:.2f} RMSE={r['rmse']:.2f} "
                    f"val_loss={r['val_loss']:.4f}\n")

    print(f"\n结果已保存至 output/phase_results.txt")


if __name__ == '__main__':
    main()
