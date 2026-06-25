import torch
import torch.nn as nn


class MultiChannelFiLM(nn.Module):
    def __init__(self, static_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim * 6)
        )
        self.latent_dim = latent_dim

    def forward(self, x_static):
        out = self.net(x_static)
        # 将输出拆分为 3 个通道各自的缩放项(g)和偏置项(b)
        soh_g, soh_b = out[:, :self.latent_dim], out[:, self.latent_dim: self.latent_dim * 2]
        pos_g, pos_b = out[:, self.latent_dim * 2: self.latent_dim * 3], out[
            :, self.latent_dim * 3: self.latent_dim * 4]
        neg_g, neg_b = out[:, self.latent_dim * 4: self.latent_dim * 5], out[:, self.latent_dim * 5:]
        return soh_g, soh_b, pos_g, pos_b, neg_g, neg_b


class AdaptiveChannelEncoder(nn.Module):
    """
    自适应通道时序编码器
    使用带有 Feature Dropout 的单层 GRU，提取特定物理特征序列的深层时序表征。
    """

    def __init__(self, input_dim: int, latent_dim: int, feat_dropout_prob: float = 0.0):
        super().__init__()
        self.feat_dropout = nn.Dropout2d(p=feat_dropout_prob)
        self.gru = nn.GRU(input_dim, latent_dim, batch_first=True)

    def forward(self, x_seq):
        x_seq = x_seq.transpose(1, 2).unsqueeze(-1)
        x_seq = self.feat_dropout(x_seq).squeeze(-1).transpose(1, 2)
        _, h = self.gru(x_seq)
        return h.squeeze(0)


class TrajectoryReconstructionDecoder(nn.Module):
    """
    轨迹重建解码器
    基于融合后的多通道特征和目标未来周期 k（归一化），重建预测出未来的 SOH 状态。
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim * 3 + 1, 64),
            nn.Softplus(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )

    def forward(self, z_soh, z_pos, z_neg, k):
        inputs = torch.cat([z_soh, z_pos, z_neg, k], dim=1)
        return self.decoder_fc(inputs)


class TAPMNetModel(nn.Module):
    """
    TAPM-Net (Temporal-Adaptive Parameter-Modulated Network) 主模型
    """

    def __init__(self, latent_dim: int, pos_features_num: int, neg_features_num: int, static_features: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.soh_encoder = AdaptiveChannelEncoder(1, latent_dim, 0.0)
        self.pos_encoder = AdaptiveChannelEncoder(pos_features_num, latent_dim, 0.0)
        self.neg_encoder = AdaptiveChannelEncoder(neg_features_num, latent_dim, 0.0)
        self.film = MultiChannelFiLM(static_features, latent_dim)
        self.decoder = TrajectoryReconstructionDecoder(latent_dim)

    def forward(self, x_history_soh, x_history_pos, x_history_neg, x_static, k_future):
        z_soh = self.soh_encoder(x_history_soh)
        z_pos = self.pos_encoder(x_history_pos)
        z_neg = self.neg_encoder(x_history_neg)

        # 获取 FiLM 调制参数
        soh_g, soh_b, pos_g, pos_b, neg_g, neg_b = self.film(x_static)

        # 调制隐空间表征
        z_soh_modulated = soh_g * z_soh + soh_b
        z_pos_modulated = pos_g * z_pos + pos_b
        z_neg_modulated = neg_g * z_neg + neg_b

        # 轨迹预测
        soh_pred = self.decoder(z_soh_modulated, z_pos_modulated, z_neg_modulated, k_future)
        return soh_pred