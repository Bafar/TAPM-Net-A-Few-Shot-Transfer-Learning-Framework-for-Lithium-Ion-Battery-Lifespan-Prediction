import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.dataset import extract_single_battery_cleaned, BatteryTAPMDataset
from src.transfer import load_model_and_scaler, transfer_and_finetune_model
from src.visualization import predict_new_battery_lifespan, plot_raw_battery_diagnostic_report

def main():
    parser = argparse.ArgumentParser(description="电池健康特征迁移与多步外推推断框架 (TAPM-Net)")
    parser.add_argument("--model_path", type=str, default="checkpoints/tapm_net_model.pth", help="预训练模型路径")
    parser.add_argument("--csv_path", type=str, required=True, help="待推断/待迁移微调的单电池原始CSV文件路径")
    parser.add_argument("--run_finetune", action="store_true", help="是否启动少样本迁移学习微调")
    parser.add_argument("--show_report", action="store_true", help="是否先显示8大退化物理量清洗报告")
    args = parser.parse_args()

    # 1. 检查物理设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 诊断报告绘制
    if args.show_report:
        plot_raw_battery_diagnostic_report(args.csv_path)

    # 3. 加载基础预训练权重及归一化器
    model, scaler, hist_w, pos_cols, neg_cols, static_cols = load_model_and_scaler(args.model_path, device)

    # 4. 执行预测 (若不进行微调，直接前向外推推演)
    if not args.run_finetune:
        predict_new_battery_lifespan(
            args.csv_path, model, scaler, hist_w, pos_cols, neg_cols, static_cols
        )
    else:
        # 执行少样本在线迁移学习微调
        df_new_cleaned = extract_single_battery_cleaned(args.csv_path)
        cell_id = df_new_cleaned['cell_id'].iloc[0]

        # 动态组装微调临时配置项
        class TempConfig:
            history_window = hist_w
            target_col = 'SOH'
            pos_exogenous_cols = pos_cols
            neg_exogenous_cols = neg_cols
            static_cols = static_cols
            pos_features_num = len(pos_cols)
            neg_features_num = len(neg_cols)
            static_features = len(static_cols)
            train_cells = [cell_id]
            test_cells = [cell_id]

        new_battery_dataset = BatteryTAPMDataset(
            df_new_cleaned, TempConfig(), is_train=False, scaler=scaler
        )
        new_battery_loader = DataLoader(new_battery_dataset, batch_size=16, shuffle=True)

        # 锁定底层表征编码，微调环境调制参数
        fine_tuned_model = transfer_and_finetune_model(
            new_battery_loader, args.model_path, epochs=30, lr=0.0005
        )

        # 基于微调后的模型生成校正后的推演曲线
        predict_new_battery_lifespan(
            args.csv_path, fine_tuned_model, scaler, hist_w, pos_cols, neg_cols, static_cols
        )

if __name__ == '__main__':
    main()