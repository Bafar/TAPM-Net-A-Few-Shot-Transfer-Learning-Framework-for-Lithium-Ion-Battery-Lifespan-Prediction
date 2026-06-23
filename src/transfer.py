import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from .models import TAPMNetModel


def load_model_and_scaler(model_path: str, device: torch.device = None):
    """
    模型装载工具：从打包的权重文件中解析配置，并重建模型与数据归一化器
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"正在读取模型工件 {model_path} ...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    history_window = checkpoint['history_window']
    latent_dim = checkpoint['latent_dim']
    pos_exogenous_cols = checkpoint['pos_exogenous_cols']
    neg_exogenous_cols = checkpoint['neg_exogenous_cols']
    static_cols = checkpoint['static_cols']

    model = TAPMNetModel(
        latent_dim=latent_dim,
        pos_features_num=len(pos_exogenous_cols),
        neg_features_num=len(neg_exogenous_cols),
        static_features=len(static_cols)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    scaler.var_ = checkpoint['scaler_scale'] ** 2

    print("基础模型和标准化标定工具装载完毕。")
    return model, scaler, history_window, pos_exogenous_cols, neg_exogenous_cols, static_cols


def transfer_and_finetune_model(new_battery_train_loader: DataLoader, pre_trained_model_path: str, epochs: int = 30,
                                lr: float = 0.0005):
    """
    少样本迁移微调策略
    1. 冻结提取通用退化表征的底层时序编码器 (Encoder)
    2. 释放上层环境调制模块 (FiLM) 与决策解码器 (Decoder) 参数
    3. 仅对可训练权重微调，实现高效率的小样本适应
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        checkpoint = torch.load(pre_trained_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(pre_trained_model_path, map_location=device)

    latent_dim = checkpoint['latent_dim']
    pos_features_num = len(checkpoint['pos_exogenous_cols'])
    neg_features_num = len(checkpoint['neg_exogenous_cols'])
    static_features = len(checkpoint['static_cols'])

    # 重建并加载预训练权重
    model = TAPMNetModel(latent_dim, pos_features_num, neg_features_num, static_features).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 【迁移学习关键步骤】冻结时序编码器，锁定通用特征映射
    print("\n[迁移微调激活] 正在锁定基础序列编码通道 (soh, pos, neg Encoders)...")
    for param in model.soh_encoder.parameters():
        param.requires_grad = False
    for param in model.pos_encoder.parameters():
        param.requires_grad = False
    for param in model.neg_encoder.parameters():
        param.requires_grad = False

    # 提取调制器与解码器的参数进行微调
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.MSELoss()

    print(f"开始在新电池上执行少样本微调 (训练轮数: {epochs})...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in new_battery_train_loader:
            h_soh = batch['x_hist_soh'].to(device)
            h_pos = batch['x_hist_pos'].to(device)
            h_neg = batch['x_hist_neg'].to(device)
            x_stat = batch['x_static'].to(device)
            k_pos = batch['k_future'].to(device)
            y_true = batch['y_future'].to(device)

            optimizer.zero_grad()
            y_pred = model(h_soh, h_pos, h_neg, x_stat, k_pos)

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  -> 微调轮次 [{epoch + 1}/{epochs}] | 平均损失: {epoch_loss / len(new_battery_train_loader):.6f}")

    print("少样本迁移微调完成。\n")
    return model