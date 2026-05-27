import torch
import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.PatchTST import Model


class PhasePredictor:
    def __init__(self, checkpoints_dir, dataset_dir):
        self.models = {}
        self.scalers = {}
        self.device = torch.device('cpu')
        phases = ['rise', 'peak', 'decline', 'trough']

        for phase in phases:
            ckpt_path = os.path.join(
                checkpoints_dir,
                f'sunspot_phase_{phase}_PatchTST_phase_{phase}_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0',
                'full_checkpoint.pth')
            if not os.path.exists(ckpt_path):
                print(f"WARNING: no checkpoint for {phase}")
                continue

            args = argparse.Namespace()
            args.enc_in = 3; args.dec_in = 3; args.c_out = 1
            args.seq_len = 96; args.pred_len = 24
            args.d_model = 512; args.n_heads = 8
            args.e_layers = 2; args.d_layers = 1; args.d_ff = 2048
            args.dropout = 0.05; args.fc_dropout = 0.05; args.head_dropout = 0.0
            args.patch_len = 16; args.stride = 8; args.padding_patch = 'end'
            args.revin = 1; args.affine = 0; args.subtract_last = 0
            args.decomposition = 0; args.kernel_size = 25; args.individual = 0
            args.embed = 'timeF'; args.activation = 'gelu'
            args.output_attention = False

            model = Model(args)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            self.models[phase] = model

            df = pd.read_csv(os.path.join(dataset_dir, f'phase_{phase}.csv'))
            scaler = StandardScaler()
            scaler.fit(df.iloc[:int(len(df)*0.76), 1:].values)
            self.scalers[phase] = scaler

        self.phases = phases
        self.seq_len = 96
        self.pred_len = 24

    def _build_input(self, ssn_array):
        n = len(ssn_array)
        months = np.arange(n) % 12 + 1
        data = np.column_stack([
            ssn_array,
            np.sin(2 * np.pi * months / 12),
            np.cos(2 * np.pi * months / 12),
        ])
        return data

    def _predict_phase(self, phase, ssn_chunk):
        if phase not in self.models:
            return np.full(self.pred_len, np.nan)

        data = self._build_input(ssn_chunk)
        scaler = self.scalers[phase]
        scaled = scaler.transform(data)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = self.models[phase](x)

        pred_scaled = out[0, :self.pred_len, 0].numpy()
        n_features = scaler.scale_.shape[0]
        dummy = np.zeros((len(pred_scaled), n_features))
        dummy[:, 0] = pred_scaled
        pred_real = scaler.inverse_transform(dummy)[:, 0]
        return pred_real

    def rolling_forecast(self, initial_ssn, start_date, n_steps=132,
                         t_high=0.57, t_low=0.11):
        current_ssn = initial_ssn.copy()
        predictions = []
        actual_dates = []
        used_phases = []

        base_date = pd.to_datetime(start_date)

        for step in range(n_steps):
            if len(current_ssn) < self.seq_len:
                predictions.append(float('nan'))
                actual_dates.append(str((base_date + pd.DateOffset(months=len(predictions))).date()))
                used_phases.append('insufficient_data')
                continue

            recent_ssn = current_ssn[-self.seq_len:]

            avg_recent = np.mean(recent_ssn[-6:])
            median_recent = np.median(recent_ssn[-12:])
            trend = np.polyfit(np.arange(12), recent_ssn[-12:], 1)[0]

            if avg_recent < 12 and trend < 1:
                phase = 'trough'
            elif trend > 4:
                phase = 'rise'
            elif avg_recent > 80 and trend >= -4:
                phase = 'peak'
            elif trend < -2:
                phase = 'decline'
            elif avg_recent < 30:
                phase = 'trough'
            else:
                phase = 'decline'

            pred = self._predict_phase(phase, recent_ssn)

            pred_val = max(0.0, float(pred[0]))
            predictions.append(pred_val)
            actual_dates.append(str((base_date + pd.DateOffset(months=len(predictions))).date()))
            used_phases.append(phase)

            current_ssn = np.append(current_ssn, pred_val)

        return np.array(predictions), np.array(actual_dates), np.array(used_phases)


import argparse

def main():
    tagged = pd.read_csv('/tmp/patchTST/phase_analysis/output/tagged_cycles_a.csv')
    tagged['date'] = pd.to_datetime(tagged['date'])
    tagged = tagged.sort_values('date').reset_index(drop=True)

    train_end_idx = int(len(tagged) * 0.85)
    test_df = tagged.iloc[train_end_idx:].reset_index(drop=True)

    initial_ssn = tagged['ssn'].values[:train_end_idx]

    start_input_date = test_df['date'].iloc[0]
    predictor = PhasePredictor(
        checkpoints_dir='./checkpoints_phase/',
        dataset_dir='./dataset_phase/',
    )

    predictions, dates, phases = predictor.rolling_forecast(
        initial_ssn, str(start_input_date.date()), n_steps=len(test_df))

    actual_ssn = test_df['ssn'].values[:len(predictions)]
    actual_dates = test_df['date'].dt.strftime('%Y-%m-%d').values[:len(predictions)]

    valid = ~np.isnan(predictions)
    if valid.sum() > 0:
        mae = np.mean(np.abs(predictions[valid] - actual_ssn[valid]))
        rmse = np.sqrt(np.mean((predictions[valid] - actual_ssn[valid])**2))
        print(f"测试样本: {valid.sum()}")
        print(f"MAE={mae:.1f}  RMSE={rmse:.1f}")
        print(f"实际SSN范围: {actual_ssn.min():.1f}~{actual_ssn.max():.1f}")
        print(f"预测SSN范围: {predictions[valid].min():.1f}~{predictions[valid].max():.1f}")

        used = pd.Series(phases[valid]).value_counts()
        print(f"各阶段使用次数: {dict(used)}")

    out = pd.DataFrame({
        'date': actual_dates[:len(predictions)],
        'actual_ssn': actual_ssn.astype(float),
        'pred_ssn': predictions.astype(float),
        'phase_used': phases,
    })
    out.to_csv('phase_training/output/rolling_prediction.csv', index=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pd.to_datetime(out['date']), out['actual_ssn'], 'k-', label='Actual', linewidth=1.5)
    ax.plot(pd.to_datetime(out['date']), out['pred_ssn'], 'r--', label='Predicted', linewidth=1)
    ax.legend()
    ax.set_title('Rolling Phase Prediction vs Actual')
    ax.set_xlabel('Date')
    ax.set_ylabel('SSN')
    plt.tight_layout()
    plt.savefig('phase_training/output/rolling_prediction.png', dpi=120)
    plt.close()
    print("图已保存: phase_training/output/rolling_prediction.png")


if __name__ == '__main__':
    main()
