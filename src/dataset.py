import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def hampel_filter_pandas(series: pd.Series, window: int = 11, n_sigmas: int = 3) -> pd.Series:
    """ 高性能向量化 Hampel 滤波器，用于过滤时序信号中的离群孤立噪声 """
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window, center=True, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = (series - rolling_median).abs()
    outlier_mask = difference > threshold
    cleaned_series = series.copy()
    cleaned_series[outlier_mask] = rolling_median[outlier_mask]
    return cleaned_series


def extract_single_battery_cleaned(csv_path: str) -> pd.DataFrame:
    """
    一键物理特征提取与三层级数据清洗
    """
    filename = os.path.basename(csv_path)
    cell_id = filename.replace('.csv', '')

    # 1. 自动解析工况元数据
    try:
        temp = float(re.search(r'CY(\d+)', filename).group(1))
        rate_str = re.search(r'CY\d+-(.+?)_', filename).group(1)
        if rate_str.startswith('0'):
            charge_rate = float(rate_str) / 10.0 if len(rate_str) == 2 else float(rate_str[0] + '.' + rate_str[1:])
        else:
            charge_rate = float(rate_str)
    except Exception:
        temp, charge_rate = 25.0, 1.0

    # 2. 读取原始电化学物理量
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    current_col, voltage_col, cycle_col, time_col = '<I>/mA', 'Ecell/V', 'cycle number', 'time/s'
    q_discharge_col = 'Q discharge/mA.h' if 'Q discharge/mA.h' in df.columns else 'Q discharge/mA. h'
    q_charge_col = 'Q charge/mA.h' if 'Q charge/mA.h' in df.columns else 'Q charge/mA. h'

    # 计算内阻 R0
    I_raw, V_raw, cycle_raw = df[current_col].values, df[voltage_col].values, df[cycle_col].values
    r0_dict = {}
    for i in range(1, len(df)):
        if I_raw[i] < -500 and I_raw[i-1] >= -10:
            delta_v = V_raw[i] - V_raw[i-1]
            delta_i = (I_raw[i] - I_raw[i-1]) / 1000.0
            if delta_i != 0:
                r0 = abs(delta_v / delta_i)
                if 0.001 < r0 < 0.5:
                    r0_dict[cycle_raw[i]] = r0

    # 3. 按循环提取 8 个特征维度
    grouped = df.groupby(cycle_col)
    q0_list = [grouped.get_group(c)[q_discharge_col].max() for c in [2, 3, 4] if c in grouped.groups]
    Q0 = np.mean(q0_list) if len(q0_list) > 0 else 3200.0

    records = []
    for cycle_num, group in grouped:
        if cycle_num < 2:
            continue
        max_disch = group[q_discharge_col].max()
        max_charge = group[q_charge_col].max()
        soh = (max_disch / Q0) * 100
        ce = (max_disch / max_charge) * 100 if max_charge > 0 else np.nan

        disch_seg = group[group[current_col] < -100]
        v_mid = disch_seg[voltage_col].median() if not disch_seg.empty else np.nan

        cv_seg = group[group['control/V'] > 4.19]
        p_cv = ((cv_seg[q_charge_col].max() - cv_seg[q_charge_col].min()) / max_charge) * 100 if not cv_seg.empty and max_charge > 0 else np.nan

        cc_seg = group[(group[current_col] > 100) & (group['control/V'] < 4.19)]
        t_cc = cc_seg[time_col].max() - cc_seg[time_col].min() if not cc_seg.empty else np.nan

        rest_seg = group[(group[current_col] == 0) & (group[q_charge_col] > 0.9 * max_charge)]
        v_relax_drop = rest_seg[voltage_col].max() - rest_seg[voltage_col].min() if not rest_seg.empty else np.nan

        cv_t_seg = group[(group['control/V'] > 4.19) & (group[current_col] > 20)]
        t_cv = cv_t_seg[time_col].max() - cv_t_seg[time_col].min() if not cv_t_seg.empty else np.nan
        r0 = r0_dict.get(cycle_num, np.nan)

        records.append({
            'cycle': cycle_num, 'SOH': soh, 'CE': ce, 'V_mid': v_mid,
            'P_CV': p_cv, 'R0': r0, 't_CC': t_cc, 'V_relax_drop': v_relax_drop, 't_CV': t_cv
        })

    df_cell = pd.DataFrame(records)

    # 4. 三层高精度物理清洗 (硬阈值剪裁、插值重组、Hampel去噪)
    df_cell.loc[(df_cell['SOH'] < 0) | (df_cell['SOH'] > 110), 'SOH'] = np.nan
    df_cell.loc[(df_cell['CE'] < 95.0) | (df_cell['CE'] > 102.0), 'CE'] = np.nan
    df_cell.loc[(df_cell['V_mid'] < 3.0) | (df_cell['V_mid'] > 3.6), 'V_mid'] = np.nan
    df_cell.loc[(df_cell['P_CV'] < 2.0) | (df_cell['P_CV'] > 60.0), 'P_CV'] = np.nan
    df_cell.loc[(df_cell['R0'] < 0.0001) | (df_cell['R0'] > 0.05), 'R0'] = np.nan
    df_cell.loc[(df_cell['t_CC'] < 500) | (df_cell['t_CC'] > 20000), 't_CC'] = np.nan
    df_cell.loc[(df_cell['V_relax_drop'] < 0.001) | (df_cell['V_relax_drop'] > 0.1), 'V_relax_drop'] = np.nan
    df_cell.loc[(df_cell['t_CV'] < 500) | (df_cell['t_CV'] > 20000), 't_CV'] = np.nan

    df_cell = df_cell.interpolate(method='linear', limit_direction='both').bfill().ffill()

    for col in ['SOH', 'CE', 'V_mid', 'P_CV', 'R0', 't_CC', 'V_relax_drop', 't_CV']:
        df_cell[col] = hampel_filter_pandas(df_cell[col], window=11, n_sigmas=3)

    df_cell = df_cell.interpolate(method='linear', limit_direction='both').bfill().ffill()

    # 5. 注入元数据
    df_cell.insert(0, 'cell_id', cell_id)
    df_cell.insert(1, 'temperature', temp)
    df_cell.insert(2, 'charge_rate', charge_rate)

    return df_cell


class BatteryTAPMDataset(Dataset):
    """
    用于 TAPM-Net 框架训练与评估的电池时序轨迹数据集
    """
    def __init__(self, df: pd.DataFrame, cfg, is_train: bool = True, scaler: StandardScaler = None):
        self.cfg = cfg
        self.samples = []

        cells = cfg.train_cells if is_train else cfg.test_cells
        self.df_sub = df[df['cell_id'].isin(cells)].copy()

        if self.df_sub.empty:
            raise ValueError("没有找到对应的电池数据，请检查输入标识符！")

        self.all_dynamic_cols = cfg.pos_exogenous_cols + cfg.neg_exogenous_cols
        if is_train:
            self.scaler = StandardScaler()
            self.df_sub[self.all_dynamic_cols] = self.scaler.fit_transform(self.df_sub[self.all_dynamic_cols].values)
        else:
            self.scaler = scaler
            self.df_sub[self.all_dynamic_cols] = self.scaler.transform(self.df_sub[self.all_dynamic_cols].values)

        for cell_id, group in self.df_sub.groupby('cell_id'):
            group = group.sort_values('cycle').reset_index(drop=True)
            if len(group) <= cfg.history_window:
                continue

            # 提取 M 圈的历史基准
            hist_group = group.iloc[:cfg.history_window]
            x_hist_soh = hist_group[[cfg.target_col]].values / 100.0
            x_hist_pos = hist_group[cfg.pos_exogenous_cols].values
            x_hist_neg = hist_group[cfg.neg_exogenous_cols].values

            # 归一化静态环境变量
            raw_static = hist_group[cfg.static_cols].iloc[0].values
            temp_scaled = (raw_static[0] - 25.0) / 20.0       # 25~45度缩放到 [0, 1]
            rate_scaled = (raw_static[1] - 0.25) / 0.75       # 0.25~1C缩放到 [0, 1]
            x_static = np.array([temp_scaled, rate_scaled])

            # 构建未来步 k 样本
            for idx in range(cfg.history_window, len(group)):
                target_row = group.iloc[idx]
                k_future = np.array([target_row['cycle'] / 1000.0])
                y_future = np.array([target_row[cfg.target_col] / 100.0])

                self.samples.append({
                    'x_hist_soh': torch.tensor(x_hist_soh, dtype=torch.float32),
                    'x_hist_pos': torch.tensor(x_hist_pos, dtype=torch.float32),
                    'x_hist_neg': torch.tensor(x_hist_neg, dtype=torch.float32),
                    'x_static': torch.tensor(x_static, dtype=torch.float32),
                    'k_future': torch.tensor(k_future, dtype=torch.float32),
                    'y_future': torch.tensor(y_future, dtype=torch.float32),
                    'cell_id': cell_id,
                    'raw_k': target_row['cycle']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]