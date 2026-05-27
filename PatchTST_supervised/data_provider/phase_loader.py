import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Phase(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', target='ssn', scale=True, timeenc=1, freq='m',
                 phase_name='rise'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 48
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.phase_name = phase_name
        self.root_path = root_path

        self._read_data()

    def _read_data(self):
        self.scaler = StandardScaler()

        data_path = os.path.join(self.root_path,
                                 f'phase_{self.phase_name}.csv')
        df_raw = pd.read_csv(data_path)

        if self.features in ('M', 'MS'):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        total = len(df_raw)
        num_test = max(1, int(total * 0.12))
        num_val = max(1, int(total * 0.12))
        num_train = total - num_val - num_test

        border1s = [0, num_train - self.seq_len, total - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, total]
        set_idx = {'train': 0, 'val': 1, 'test': 2}[self.flag]
        border1 = border1s[set_idx]
        border2 = border2s[set_idx]

        if self.scale:
            train_data = df_data.values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']].iloc[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values),
                                       freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        eff_len = max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
        print(f"[Dataset_Phase] {self.phase_name}/{self.flag}: "
              f"total={total} slice=[{border1}:{border2}]={border2-border1}  "
              f"samples={eff_len}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def phase_data_provider(args, flag, phase_name):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Dataset_Phase(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        phase_name=phase_name,
    )
    print(f"{flag}/{phase_name} {len(data_set)}")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
