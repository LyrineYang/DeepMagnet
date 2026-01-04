# DeepMagnet: 抗干扰电磁 CT 逆向成像系统

> **DeepMagnet** 是一个基于深度学习的高精度电磁层析成像 (EMT) 系统。它利用 GridSegNet 架构，在极低信噪比的矿化土壤环境下，实时还原地下金属目标的 3D 形状与位置。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)
![CUDA](https://img.shields.io/badge/CUDA-12.6-green)

---

## 📖 项目简介

传统的金属探测器只能听到"滴滴"声，而 **DeepMagnet** 给它装上了"眼睛"。

1. **物理仿真**：模拟双 D 线圈 (Double-D) 在 24×24 网格轨迹下的感应电压信号
2. **AI 重建**：利用 GridSegNet (Conv2D Encoder + 3D Decoder) 从 2D 热力图逆向重构 64³ 体素分布
3. **抗干扰演示**：展示 AI 如何在强土壤矿化噪声中提取有效特征

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🚀 **实时推理** | 单卡推理 < 50ms，输出 64×64×64 体素 |
| 🎯 **高精度** | 过拟合测试 Dice > 0.82 |
| 📊 **2D→3D** | 24×24 信号热力图 → 64³ 3D 体素重建 |
| 🔧 **模块化** | 数据生成、模型、推理严格解耦 |

---

## 🏗️ 模型架构

```
输入: (B, 4, 24, 24)         输出: (B, 64, 64, 64)
      ↓                              ↑
  ┌─────────────────┐      ┌─────────────────┐
  │  Conv2D Encoder │ ──→  │  3D Decoder     │
  │  (4→64→128→256) │      │  (FC→TransConv) │
  │  + Residual     │      │  64→32→16→1     │
  └─────────────────┘      └─────────────────┘
       37.5M 参数
```

**输入通道说明**：
- Channel 0: 信号强度 (Log+MinMax 归一化)
- Channel 1-3: XYZ 位置编码

---

## 🛠️ 安装指南

```bash
# 创建环境
conda create -n deepmagnet python=3.10 -y && conda activate deepmagnet

# 安装 PyTorch (CUDA 12.x)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt

# 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 快速上手

### 1. 生成数据
```bash
python -m src.data.generate --config configs/data_overfit.yaml
```

### 2. 训练模型
```bash
python -m src.trainers.train \
  --data configs/data_overfit.yaml \
  --model configs/model.yaml \
  --train configs/train_overfit.yaml
```

### 3. 推理
```python
from src.models.inference import load_model, run_inference

# 加载模型
model = load_model('outputs/checkpoints_overfit/best_model.pt', device='cuda')

# 推理 (输入 4×24×24 预处理信号)
volume = run_inference(model, preprocessed_signal)
# 输出: (64, 64, 64) 概率图
```

---

## 📂 项目结构

```
DeepMagnet/
├── configs/                    # 配置文件
│   ├── data_overfit.yaml       # 数据配置 (576步, 24×24网格)
│   ├── model.yaml              # 模型配置 (GridSegNet)
│   └── train_overfit.yaml      # 训练配置
├── src/
│   ├── data/
│   │   ├── dataset.py          # 数据集 (自动预处理)
│   │   ├── generate.py         # 数据生成器
│   │   └── shapes.py           # 3D形状生成
│   ├── models/
│   │   ├── grid_segnet.py      # GridSegNet 模型
│   │   ├── losses.py           # 损失函数 (BCE+Dice)
│   │   └── inference.py        # 推理接口
│   └── trainers/
│       └── train.py            # 训练脚本
├── tests/                      # 过拟合测试脚本
├── scripts/                    # 工具脚本
└── requirements.txt
```

---

## ⚙️ 关键配置

### model.yaml
```yaml
arch:
  name: grid_segnet
  input_grid_size: 24
  grid_encoder:
    base_channels: 64
    depth: 3
    latent_dim: 512
  decoder3d:
    base_channels: 64
    depth: 4

loss:
  mask:
    type: dice_bce
    pos_weight: 20.0  # 正样本权重 (稀疏目标)
```

### 训练参数
| 参数 | 值 |
|------|-----|
| Epochs | 500 |
| Batch Size | 5 |
| Learning Rate | 1e-3 → 1e-6 (cosine) |
| pos_weight | 20.0 |
| Loss | 0.5×BCE + 0.5×Dice |

---

## 📈 训练结果

过拟合测试 (45 样本, 500 epochs):
```
Epoch   0: train_dice=0.13
Epoch 100: train_dice=0.72
Epoch 499: train_dice=0.97, best_dice=0.46 (val)
```

推理测试:
```
Dice Score: 0.826 ✅
```

---

## 📜 许可证

MIT License
