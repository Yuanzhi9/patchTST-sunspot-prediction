import pandas as pd
import numpy as np
import os

TAGGED_CSV = '../tagged_cycles_a.csv'
OUTPUT_DIR = '../PatchTST_supervised/dataset_phase'

PHASE_LABELS = ['rise', 'peak', 'decline', 'trough']


def build_phase_datasets():
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))
    else:
        os.makedirs(OUTPUT_DIR)

    df = pd.read_csv(TAGGED_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

    print(f"标记数据: {len(df)}行, {df['cycle'].nunique()}个周期")

    for phase in PHASE_LABELS:
        phase_df = df[df['phase'] == phase].copy()
        phase_df = phase_df.sort_values(['cycle', 'date']).reset_index(drop=True)
        cols = ['date', 'month_sin', 'month_cos', 'ssn']
        phase_df[cols].to_csv(f'{OUTPUT_DIR}/phase_{phase}.csv', index=False)
        print(f"  {phase:8s}: {len(phase_df):4d}行 → {OUTPUT_DIR}/phase_{phase}.csv")

    print(f"\n输出: {OUTPUT_DIR}/")


if __name__ == '__main__':
    build_phase_datasets()
