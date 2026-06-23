# TAPM-Net: 基于少样本迁移微调的锂电池寿命预测框架

这是一个面向锂离子电池（Lithium-ion Battery）健康状态（SOH）预测与终期寿命（EOL）多步外推推断的 PyTorch 项目。承接上一个项目，本项目仅仅实现预测模型从电池源域迁移到未知的目标域。

---

## 迁移学习的核心思想与方法

在不同环境温度与充放电倍率下，锂电池的退化机理虽然具有变异性，但依然遵循相似的电化学物理演化规律。

### 1. 通用电化学特征编码器的“参数冻结” (Parameter Freezing)
- **原理**：我们在模型底层设计了 3 个平行的 `AdaptiveChannelEncoder`（基于 GRU 架构），分别提取 SOH 自相关序列、正极相关特征、负极相关特征的时序关联。
- **迁移实现**：迁移学习的核心假设是底层通用时序特征映射是跨工况共享的。因此，在目标域少样本微调阶段，我们将这三个通道的时序编码器参数设为 `requires_grad = False`，完全冻结其权重。这能有效保留源域大规模训练中提炼出的稳健退化物理表征，防止新电池在极少周期样本上过拟合。

### 2. 环境特征条件化：多通道 FiLM 参数调制 (Multi-Channel FiLM Modulation)
- **原理**：特征线性调制（Feature-wise Linear Modulation, FiLM）用于自适应融合电化学物理信息与环境条件。我们使用 `MultiChannelFiLM` 结构，将连续变换的静态特征（缩放后的温度 Temp 与充电倍率 C-Rate）传入小型网络，动态预测出一组缩放项（Scale, $\gamma$）和偏置项（Shift, $\beta$）。
- **调制方法**：对冻结编码器提取出的潜在空间向量 $Z$ 执行仿射变换：
  $$Z_{\text{modulated}} = \gamma \odot Z + \beta$$
  该操作可以在不改变底层物理表征提取通道的情况下，通过小规模参数平移，动态校正不同温度、倍率带来的退化斜率和残余偏置差异。

### 3. 少样本微调策略 (Few-Shot Fine-Tuning)
- 当我们拿到一块仅有极初期运行数据的未知电池时，通过冻结时序层、仅释放上层环境调制器 `FiLM` 以及后端多步融合解码器 `Decoder`，在极低的参数学习空间（Few-Shot）中，利用小样本快速完成针对特定电池个体差异的适配和对齐。

---

##  项目依赖
请在终端中执行以下命令安装依赖：
```bash
pip install -r requirements.txt
```

---

## 快速上手说明

### 1. 运行一键物理指标清洗诊断
当你想查看一块全新电池原始 CSV 中提取清洗后的 8 个关键物理退化指标曲线时，可运行：
```bash
python main.py --csv_path "data/YOUR_BATTERY.csv" --show_report
```

### 2. 纯零样本前向外推推断
若要直接加载预训练模型（不经过任何参数调校），推演全新电池的寿命轨迹：
```bash
python main.py --model_path "checkpoints/tapm_net_model.pth" --csv_path "data/YOUR_BATTERY.csv"
```

### 3. 少样本迁移学习与微调推演
若要在新电池的前 $M$ 圈已知寿命序列上对模型执行内存迁移微调，校正后续数百圈的连续退化路径：
```bash
python main.py --model_path "checkpoints/tapm_net_model.pth" --csv_path "data/YOUR_BATTERY.csv" --run_finetune
```
