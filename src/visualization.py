import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from .dataset import extract_single_battery_cleaned

# 配置中文支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

def predict_new_battery_lifespan(new_csv_path: str, model, scaler, history_window: int,
                                 pos_exogenous_cols, neg_exogenous_cols, static_cols,
                                 eol_threshold: float = 80.0):
    """
    一键评估与寿命推演绘图
    """
    filename = os.path.basename(new_csv_path)
    cell_id = filename.replace('.csv', '')
    device = next(model.parameters()).device

    print(f"\n评估新电池退化演化: {cell_id}")

    try:
        temp = float(re.search(r'CY(\d+)', filename).group(1))
        rate_str = re.search(r'CY\d+-(.+?)_', filename).group(1)
        if rate_str.startswith('0'):
            charge_rate = float(rate_str) / 10.0 if len(rate_str) == 2 else float(rate_str[0] + '.' + rate_str[1:])
        else:
            charge_rate = float(rate_str)
    except Exception:
        temp, charge_rate = 25.0, 1.0

    temp_scaled = (temp - 25.0) / 20.0
    rate_scaled = (charge_rate - 0.25) / 0.75
    x_static_tensor = torch.tensor(np.array([temp_scaled, rate_scaled]), dtype=torch.float32).unsqueeze(0).to(device)

    # 提取特征
    df = pd.read_csv(new_csv_path)
    df.columns = df.columns.str.strip()
    current_col, voltage_col, cycle_col, time_col = '<I>/mA', 'Ecell/V', 'cycle number', 'time/s'
    q_discharge_col = 'Q discharge/mA.h' if 'Q discharge/mA.h' in df.columns else 'Q discharge/mA. h'
    q_charge_col = 'Q charge/mA.h' if 'Q charge/mA.h' in df.columns else 'Q charge/mA. h'

    # 估算基准 Q0
    grouped = df.groupby(cycle_col)
    q0_list = [grouped.get_group(c)[q_discharge_col].max() for c in [2, 3, 4] if c in grouped.groups]
    Q0 = np.mean(q0_list) if len(q0_list) > 0 else 3200.0

    # 提取 R0 字典
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

    records, actual_cycles, actual_soh = [], [], []
    for cycle_num, group in grouped:
        if cycle_num < 2:
            continue
        max_disch = group[q_discharge_col].max()
        max_charge = group[q_charge_col].max()
        soh = (max_disch / Q0) * 100
        ce = (max_disch / max_charge) * 100 if max_charge > 0 else np.nan

        if soh >= 50 and ce >= 90:
            actual_cycles.append(cycle_num)
            actual_soh.append(soh)

        if cycle_num <= history_window + 1:
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

            records.append({'SOH': soh, 'CE': ce, 'V_mid': v_mid, 'P_CV': p_cv, 'R0': r0, 't_CC': t_cc, 'V_relax_drop': v_relax_drop, 't_CV': t_cv})

    df_hist = pd.DataFrame(records).interpolate(method='linear', limit_direction='both').bfill().ffill()

    all_dynamic_cols = pos_exogenous_cols + neg_exogenous_cols
    df_hist[all_dynamic_cols] = scaler.transform(df_hist[all_dynamic_cols].values)

    # 模型推理准备
    x_hist_soh = torch.tensor(df_hist[['SOH']].values[:history_window] / 100.0, dtype=torch.float32).unsqueeze(0).to(device)
    x_hist_pos = torch.tensor(df_hist[pos_exogenous_cols].values[:history_window], dtype=torch.float32).unsqueeze(0).to(device)
    x_hist_neg = torch.tensor(df_hist[neg_exogenous_cols].values[:history_window], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        z_soh = model.soh_encoder(x_hist_soh)
        z_pos = model.pos_encoder(x_hist_pos)
        z_neg = model.neg_encoder(x_hist_neg)
        soh_g, soh_b, pos_g, pos_b, neg_g, neg_b = model.film(x_static_tensor)
        z_soh_m = soh_g * z_soh + soh_b
        z_pos_m = pos_g * z_pos + pos_b
        z_neg_m = neg_g * z_neg + neg_b

    projected_cycles = []
    projected_soh = []
    pred_eol = None

    for k in range(history_window + 1, 2000):
        with torch.no_grad():
            k_tensor = torch.tensor([[k / 1000.0]], dtype=torch.float32).to(device)
            pred_val = model.decoder(z_soh_m, z_pos_m, z_neg_m, k_tensor).item() * 100.0

        projected_cycles.append(k)
        projected_soh.append(pred_val)

        if pred_val <= eol_threshold and pred_eol is None:
            pred_eol = k
            max_plot_cycle = k + 100
        if pred_eol is not None and k >= max_plot_cycle:
            break

    print(f"  -> 测试条件: 环境温度 {temp} ℃ | 充电倍率 {charge_rate} C")
    if pred_eol:
        print(f"  -> 预测电池将在第 {pred_eol} 圈时跌破 {eol_threshold}% SOH 并终止寿命。")

    # 绘图
    plt.figure(figsize=(10, 5), dpi=120)
    if len(actual_cycles) > history_window:
        plt.plot(actual_cycles, actual_soh, label='实际观测轨迹 (Actual Record)', color='#4A5568', linestyle='--', linewidth=1.2, alpha=0.8)
    plt.plot(projected_cycles, projected_soh, label='PIM-TN 连续推演轨迹', color='#E53E3E', linewidth=1.6)
    plt.axhline(eol_threshold, color='#718096', linestyle=':', linewidth=1.0, alpha=0.7, label=f'EOL界限 ({eol_threshold}%)')

    if pred_eol:
        plt.scatter(pred_eol, eol_threshold, color='#E53E3E', edgecolors='white', s=80, linewidths=1.5, zorder=5, label=f'预测 EOL: {pred_eol} 圈')

    plt.title(f'电池SOH健康演变图 - {cell_id}', fontsize=12, fontweight='bold', color='#2D3748', pad=15)
    plt.xlabel('Cycle Number', fontsize=10, color='#4A5568')
    plt.ylabel('SOH (%)', fontsize=10, color='#4A5568')
    plt.grid(True, linestyle=':', alpha=0.5, color='#CBD5E0')
    plt.legend(loc='lower left', frameon=True, facecolor='#F7FAFC', edgecolor='#E2E8F0', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_raw_battery_diagnostic_report(new_csv_path: str, window: int = 11, n_sigmas: int = 3):
    """
    绘制原始未清洗数据的 8 通道特征诊断大图
    """
    filename = os.path.basename(new_csv_path)
    cell_id = filename.replace('.csv', '')

    # 直接复用清洗逻辑
    df_cell = extract_single_battery_cleaned(new_csv_path)
    temp = df_cell['temperature'].iloc[0]
    charge_rate = df_cell['charge_rate'].iloc[0]
    cycles = df_cell['cycle'].values

    fig, axs = plt.subplots(4, 2, figsize=(16, 18))
    axs = axs.ravel()

    features_config = [
        ('SOH', '健康状态 SOH Decline (%) - Cleaned', 'blue', False),
        ('CE', '库仑效率 Coulombic Efficiency (CE, %) - Cleaned', 'green', False),
        ('V_mid', '放电中段电压 Discharge Mid-point Voltage (V_mid, V) - Cleaned', 'orange', False),
        ('P_CV', '恒压充电容量占比 CV Charge Capacity Ratio (P_CV, %)', 'red', False),
        ('R0', '欧姆内阻 Ohmic Resistance (R0, Ohm)', 'purple', True),
        ('t_CC', '恒流充电时长 CC Charge Time (t_CC, seconds)', 'navy', False),
        ('V_relax_drop', '电压弛豫跌落值 Voltage Relaxation Drop (V_relax_drop, V)', 'teal', False),
        ('t_CV', '恒压充电时长 CV Charge Time (t_CV, seconds)', 'magenta', False)
    ]

    for idx, (col_name, title, color, draw_ma) in enumerate(features_config):
        y_values = df_cell[col_name].values

        if draw_ma:
            axs[idx].scatter(cycles, y_values, color=color, alpha=0.3, s=12, label='Raw calculated R0')
            smooth_y = pd.Series(y_values).rolling(window=5, min_periods=1).mean()
            axs[idx].plot(cycles, smooth_y, color='red', linewidth=1.5, label='5-cycle MA')
            axs[idx].legend(loc='upper left')
        else:
            axs[idx].plot(cycles, y_values, color=color, linewidth=1.5)

        axs[idx].set_title(title)
        axs[idx].grid(True)
        axs[idx].set_xlabel('Cycle Number')

    plt.suptitle(f'Cleaned 8-Variables Diagnostic Report of {cell_id} ({temp}℃, {charge_rate}C)', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()