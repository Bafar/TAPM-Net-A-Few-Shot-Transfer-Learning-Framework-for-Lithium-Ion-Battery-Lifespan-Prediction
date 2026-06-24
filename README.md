# TAPM-Net: 基于少样本迁移微调的锂电池寿命预测框架

这是一个面向锂离子电池（Lithium-ion Battery）健康状态（SOH）预测与终期寿命（EOL）多步外推推断的 PyTorch 项目。承接上一个项目，本项目仅仅实现预测模型从电池源域迁移到未知的目标域。
如下是未使用迁移方法在一个未知的电池上进行推演电池寿命退化的结果，尽管预测的退化路径与真实退化路径很相似，但仍有提升空间
<img width="1187" height="587" alt="image" src="https://github.com/user-attachments/assets/70dc2cce-c3b8-4904-b145-4664f8c4e279" />
如下是使用了我们提出的迁移方法，进行推演的电池寿命退化的结果，可以看到预测的准确率进一步提升，更接近真实寿命退化路径
<img width="1187" height="587" alt="image" src="https://github.com/user-attachments/assets/005a1bec-33b8-4cfc-9029-3ab16048c4c3" />
在已知的电池数据上，电池的最大放电容量尚且没有达到退化标准，该程序也能完成外推，预测未来的健康状态退化轨迹，如下所示
<img width="1188" height="587" alt="image" src="https://github.com/user-attachments/assets/120b22ad-04e0-4afb-9124-9fbe57564813" />

---

## 迁移学习的核心思想与方法

在不同环境温度与充放电倍率下，锂电池的退化机理虽然具有变异性，但依然遵循相似的电化学物理演化规律。

### 1. 通用电化学特征编码器的参数冻结
- **原理**：我们基于GRU在模型底层设计了 3 个平行的 `AdaptiveChannelEncoder`，分别提取SOH自相关序列、正相关特征、负相关特征的时序关联，将三种特征存储在静态特征向量中方便在没有除SOH外的其它电池数据的情况下进行外推。
- **迁移实现**：迁移学习的核心假设是底层通用时序特征映射是跨工况共享的。因此，在目标域少样本微调阶段，我们将这三个通道的时序编码器参数设为 `requires_grad = False`，完全冻结其权重。这能有效保留源域大规模训练中提炼出的稳健退化物理表征，防止新电池在极少周期样本上过拟合。如果充电轮数，SOH自相关序列、正相关特征、负相关特征可以分别表示为 $k$ , $z_{SOH}$ , $z_{pos}$ , $z_{neg}$ ，那么外推函数可以表示为
$$SOH(k)=f(k;z_{SOH},z_{pos},z_{neg})$$
- 外推未来时三个向量是固定不变的，这表示了电池的化学标签作为控制参数，而轮数位置是自变量。

### 2. 环境特征条件化：多通道FiLM参数调制
- **原理**：特征线性调制（Feature-wise Linear Modulation, FiLM）用于自适应融合电化学物理信息与环境条件。我们使用 `MultiChannelFiLM` 结构，将连续变换的静态特征（缩放后的温度Temp与充电倍率C-Rate）传入小型网络，动态预测出一组缩放项（Scale, $\gamma$）和偏置项（Shift, $\beta$），借助这两项进行源域和目标域的对齐。
- **调制方法**：对冻结编码器提取出的潜在空间向量 $Z$ 执行仿射变换：
  $$Z_{\text{modulated}} = \gamma \odot Z + \beta$$
  该操作可以在不改变底层物理表征提取通道的情况下，通过小规模参数平移，动态校正不同温度、倍率带来的退化斜率和残余偏置差异。

### 3. 少样本微调
- 当拿到一块仅有极初期运行数据的未知电池时，通过冻结时序层、仅释放上层环境调制器 `FiLM` 以及后端多步融合解码器 `Decoder`，在极低的学习率下，利用目标电池的少量样本（设置为至少为30轮）快速完成针对特定电池个体差异的适配和对齐，再存储为如上所述的3个$z$向量作为化学标签，再进行外推。

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
