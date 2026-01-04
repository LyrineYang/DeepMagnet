# DeepMagnet: 抗干扰电磁 CT 逆向成像系统

> **DeepMagnet** 是一个基于 DeepONet 的高精度电磁层析成像 (EMT) 演示系统。它利用深度学习技术，在极低信噪比的矿化土壤环境下，实时还原地下金属目标的 3D 形状与位置。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)
![CUDA](https://img.shields.io/badge/CUDA-12.6-green)

---

## 📖 项目简介

传统的金属探测器只能听到"滴滴"声，而 **DeepMagnet** 给它装上了"眼睛"。
1. **物理仿真**：模拟双 D 线圈 (Double-D) 和单线圈在不同轨迹下的感应电压信号。
2. **AI 重建**：利用 DeepONet / Seq-to-Vol 模型，直接从一维时序信号逆向重构 3D 磁导率分布。
3. **抗干扰演示**：交互式控制台，展示 AI 如何在强土壤矿化噪声中提取有效特征。

## ✨ 核心特性

- **🚀 实时成像**：支持 4x H100 多卡并行训练，单卡推理 < 50ms
- **🎛️ 交互式控制台**：Streamlit Web 界面，支持实时调节噪声、手写输入
- **✍️ 手写泛化测试**：模型可重建训练集中未见过的手写数字
- **🔧 模块化设计**：数据生成、模型、可视化严格解耦

---

## 🛠️ 安装指南

```bash
# 创建环境 + 安装依赖 (CUDA 12.6)
conda create -n deepmagnet python=3.10 -y && conda activate deepmagnet
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

> **注意**: PyTorch cu121 兼容 CUDA 12.6。首次数据生成会自动下载 MNIST。

---

## 🚀 快速上手

### 1. Smoke Test (验证环境)
```bash
bash scripts/smoke_test.sh  # ~1 分钟
```

### 2. 启动 Web Demo
```bash
streamlit run src/viz/web_demo.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📊 训练流程 (4x H100)

### 预计耗时

| 阶段 | 命令 | 10K 样本 |
|------|------|----------|
| **数据生成** | `python scripts/gen_data.py --config configs/data_h100.yaml --device cuda` | ~3-5 分钟 |
| **训练 (4卡)** | `torchrun --nproc_per_node=4 scripts/train_ddp.py` | ~5-10 分钟 (50 epochs) |

### 完整训练命令

```bash
# Step 1: 生成数据
python scripts/gen_data.py --config configs/data_h100.yaml --device cuda

# Step 2: 4卡 DDP 训练
torchrun --nproc_per_node=4 scripts/train_ddp.py \
  --data configs/data_h100.yaml \
  --model configs/model.yaml \
  --train configs/train_4gpu.yaml

# Step 3: 评估
python -m src.trainers.eval --ckpt outputs/checkpoints/best_model.pth --split_dir data/processed/val
```

### 配置说明

| 配置文件 | 用途 |
|----------|------|
| `configs/data_h100.yaml` | 10K 样本, 64³ 网格 |
| `configs/train_4gpu.yaml` | batch=128/卡, 16 workers |
| `configs/data_tiny.yaml` | 测试用, 8 样本 |

---

## 📂 仓库结构

```
├── configs/            # 数据/模型/训练配置
├── src/
│   ├── data/           # 物理仿真: 线圈、磁场、形状
│   ├── models/         # DeepONet, Seq-to-Vol
│   ├── trainers/       # 训练/评估脚本
│   └── viz/            # Streamlit Web Demo
├── scripts/            # 数据生成, DDP 训练
└── outputs/            # 模型权重 (自动生成)
```

---

## 📜 许可证
MIT License
