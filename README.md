# 项目说明

本仓库提供故障诊断特征提取、跨频率验证与迁移学习训练脚本。

## 1) 生成跨频率验证数据

默认：训练=12kHz_DE_data，验证=48kHz_DE_data。你可以在 `preprocess/cross_freq_split.py` 顶部修改 `TRAIN_DIR` 与 `VAL_DIR`。

运行：

```fish
python preprocess/cross_freq_split.py
```

输出 `data/cross_freq_features.npz`，包含：

- X_train, y_train：训练频率特征与标签
- X_val, y_val：验证频率特征与标签
- feature_names：特征名对齐

## 2) 训练基线模型

`model/base.py` 会优先加载 `data/cross_freq_features.npz`；若不存在则退回使用 `data/pca_features_labels.npz`（同频率随机切分）。

```fish
python model/base.py
```

## 3) 迁移学习（含目标域）

先预处理源/目标域并进行 CORAL 对齐：

```fish
python preprocess/transfer_data.py
```

然后训练 DANN 模型：

```fish
python model/transfer.py
```

日志中会打印：

- Target samples（目标域样本数）
- Batches -> source/target/val（各数据加载器批次数）

这可用于确认目标域数据已被实际使用。

## 4) 依赖

见 `pyproject.toml`。使用 uv/pip 安装均可。
